"""RemoteEnvPool: gRPC-based distributed environment pool.

Presents the same interface as :class:`~rlox.gym_vec_env.GymVecEnv` and
:class:`~rlox.VecEnv` so it can be used directly with
:class:`~rlox.collectors.RolloutCollector`.

Supports a configurable backend:
- ``grpcio``: uses the ``grpc`` package (default when available)
- ``socket``: lightweight fallback for testing without gRPC dependency

The proto schema is defined in ``crates/rlox-grpc/proto/env.proto``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class _WorkerHandle:
    """Connection handle for a single gRPC worker."""

    address: str
    channel: Any = None
    stub: Any = None


class RemoteEnvPool:
    """Pool of remote environment workers connected via gRPC.

    Presents the same interface as GymVecEnv / VecEnv so it can be used
    directly with RolloutCollector.

    Example::

        pool = RemoteEnvPool(addresses=["gpu-node-1:50051", "gpu-node-2:50051"])
        pool.connect()
        obs = pool.reset_all()
        result = pool.step_all(actions)

    Parameters
    ----------
    addresses : list[str]
        List of ``host:port`` addresses for gRPC environment workers.
    timeout : float
        Connection and call timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        addresses: list[str],
        timeout: float = 30.0,
    ) -> None:
        if not addresses:
            raise ValueError("at least one address must be provided")

        self._addresses = list(addresses)
        self._timeout = timeout
        self._connected = False
        self._workers: list[_WorkerHandle] = []
        self._num_envs = 0
        self._obs_dim = 0

        # Overridable implementations for mock injection
        self._step_impl: Any = None
        self._reset_impl: Any = None

    def connect(self) -> None:
        """Establish gRPC connections to all workers.

        Raises
        ------
        ImportError
            If ``grpcio`` is not installed.
        ConnectionError
            If any worker is unreachable.
        """
        try:
            import grpc
        except ImportError as exc:
            raise ImportError(
                "grpcio is required for RemoteEnvPool. "
                "Install it with: pip install grpcio grpcio-tools"
            ) from exc

        workers: list[_WorkerHandle] = []
        total_envs = 0

        for addr in self._addresses:
            channel = grpc.insecure_channel(
                addr,
                options=[
                    ("grpc.connect_timeout_ms", int(self._timeout * 1000)),
                ],
            )

            # Try to build a stub.  We generate minimal stub classes inline
            # to avoid requiring grpcio-tools codegen at import time.
            stub = _build_env_stub(channel)

            # Query spaces to validate connectivity and learn env count
            try:
                spaces_resp = stub.GetSpaces(
                    _empty_pb(), timeout=self._timeout
                )
            except grpc.RpcError as rpc_err:
                raise ConnectionError(
                    f"Failed to connect to worker at {addr}: {rpc_err}"
                ) from rpc_err

            num_envs = spaces_resp.num_envs
            total_envs += num_envs

            if self._obs_dim == 0:
                obs_space = json.loads(spaces_resp.obs_space_json)
                if "dim" in obs_space:
                    self._obs_dim = obs_space["dim"]

            workers.append(_WorkerHandle(
                address=addr, channel=channel, stub=stub,
            ))

        self._workers = workers
        self._num_envs = total_envs
        self._connected = True

    def _require_connection(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise ConnectionError(
                "RemoteEnvPool is not connected. Call connect() first, "
                "or ensure gRPC workers are running."
            )

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all remote environments.

        Parameters
        ----------
        actions : array-like
            Actions for each environment, shape ``(num_envs,)`` for discrete
            or ``(num_envs, act_dim)`` for continuous.

        Returns
        -------
        dict with keys ``obs``, ``rewards``, ``terminated``, ``truncated``,
        ``terminal_obs`` -- matching the ``VecEnv`` / ``GymVecEnv`` contract.
        """
        self._require_connection()

        # Allow mock injection for testing
        if self._step_impl is not None:
            return self._step_impl(actions)

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        # Determine if discrete or continuous
        discrete = actions.ndim == 1 or (actions.ndim == 2 and actions.shape[1] == 1)
        act_dim = 1 if discrete else actions.shape[1]
        flat_actions = actions.flatten().astype(np.float32).tolist()

        # Split actions across workers
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_terminal_obs: list[np.ndarray | None] = []

        offset = 0
        for worker in self._workers:
            worker_envs = _get_worker_envs(worker, self._num_envs, len(self._workers))
            worker_actions = flat_actions[offset * act_dim:(offset + worker_envs) * act_dim]
            offset += worker_envs


            step_req = worker.stub._step_request_class(
                actions=worker_actions,
                discrete=discrete,
                act_dim=act_dim,
            )
            resp = worker.stub.StepBatch(step_req, timeout=self._timeout)

            obs_dim = resp.obs_dim
            num_envs = resp.num_envs
            obs = np.array(resp.obs, dtype=np.float32).reshape(num_envs, obs_dim)
            rewards = np.array(resp.rewards, dtype=np.float64)
            terminated = np.array(resp.terminated, dtype=np.uint8)
            truncated = np.array(resp.truncated, dtype=np.uint8)

            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_terminal_obs.extend([None] * num_envs)

        return {
            "obs": np.concatenate(all_obs, axis=0),
            "rewards": np.concatenate(all_rewards),
            "terminated": np.concatenate(all_terminated),
            "truncated": np.concatenate(all_truncated),
            "terminal_obs": all_terminal_obs,
        }

    def reset_all(self, seed: int | None = None) -> np.ndarray:
        """Reset all remote environments.

        Returns
        -------
        np.ndarray of shape ``(num_envs, obs_dim)`` with dtype float32.
        """
        self._require_connection()

        # Allow mock injection for testing
        if self._reset_impl is not None:
            return self._reset_impl(seed)

        all_obs = []

        for worker in self._workers:
            reset_req = worker.stub._reset_request_class(
                seed=seed if seed is not None else 0,
                has_seed=seed is not None,
            )
            resp = worker.stub.ResetBatch(reset_req, timeout=self._timeout)

            obs_dim = resp.obs_dim
            num_envs = resp.num_envs
            obs = np.array(resp.obs, dtype=np.float32).reshape(num_envs, obs_dim)
            all_obs.append(obs)

        self._obs_dim = obs_dim
        return np.concatenate(all_obs, axis=0)

    @property
    def num_envs(self) -> int:
        """Return the total number of environments across all workers."""
        return self._num_envs

    @property
    def worker_addresses(self) -> list[str]:
        """Return the list of worker addresses."""
        return list(self._addresses)

    def close(self) -> None:
        """Close all gRPC channels."""
        for worker in self._workers:
            if worker.channel is not None:
                try:
                    worker.channel.close()
                except Exception:
                    pass
        self._workers = []
        self._connected = False

    def __repr__(self) -> str:
        return (
            f"RemoteEnvPool(addresses={len(self._addresses)}, "
            f"num_envs={self._num_envs}, "
            f"connected={self._connected})"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Internal gRPC helpers — avoid requiring grpc codegen
# ---------------------------------------------------------------------------


def _get_worker_envs(
    worker: _WorkerHandle, total_envs: int, num_workers: int
) -> int:
    """Estimate envs per worker (equal split)."""
    return total_envs // num_workers


def _build_env_stub(channel: Any) -> Any:
    """Build a minimal gRPC stub for the EnvService without codegen.

    Uses the low-level ``grpc.channel_unary_unary`` API with manual
    protobuf serialization matching ``env.proto``.
    """

    # We use grpc.protos_and_services if available, otherwise manual stubs.
    # For maximum compatibility, build manual unary-unary callables.

    step_method = channel.unary_unary(
        "/rlox.env.EnvService/StepBatch",
        request_serializer=_StepRequest.SerializeToString,
        response_deserializer=_StepResponse.FromString,
    )
    reset_method = channel.unary_unary(
        "/rlox.env.EnvService/ResetBatch",
        request_serializer=_ResetRequest.SerializeToString,
        response_deserializer=_ResetResponse.FromString,
    )
    spaces_method = channel.unary_unary(
        "/rlox.env.EnvService/GetSpaces",
        request_serializer=_Empty.SerializeToString,
        response_deserializer=_SpacesResponse.FromString,
    )

    class _Stub:
        StepBatch = step_method
        ResetBatch = reset_method
        GetSpaces = spaces_method
        _step_request_class = _StepRequest
        _reset_request_class = _ResetRequest

    return _Stub()


def _empty_pb() -> _Empty:
    """Return a serializable empty message."""
    return _Empty()


# ---------------------------------------------------------------------------
# Minimal protobuf-compatible message classes (no codegen required)
# ---------------------------------------------------------------------------
# These implement just enough of the protobuf wire format to talk to the
# Rust tonic server.  For production use with complex protos, run
# ``python -m grpc_tools.protoc`` to generate full stubs.


class _Empty:
    """Empty protobuf message."""

    @staticmethod
    def SerializeToString() -> bytes:
        return b""

    @staticmethod
    def FromString(data: bytes) -> _Empty:
        return _Empty()


class _StepRequest:
    """Manual serialization for StepRequest."""

    def __init__(
        self,
        actions: list[float] | None = None,
        discrete: bool = True,
        act_dim: int = 1,
    ) -> None:
        self.actions = actions or []
        self.discrete = discrete
        self.act_dim = act_dim

    def SerializeToString(self) -> bytes:
        """Encode as protobuf wire format."""
        parts: list[bytes] = []
        # field 1: repeated float actions (packed)
        if self.actions:
            payload = b"".join(struct.pack("<f", a) for a in self.actions)
            parts.append(_encode_tag(1, 2))  # length-delimited
            parts.append(_encode_varint(len(payload)))
            parts.append(payload)
        # field 2: bool discrete
        if self.discrete:
            parts.append(_encode_tag(2, 0))
            parts.append(_encode_varint(1))
        # field 3: uint32 act_dim
        if self.act_dim:
            parts.append(_encode_tag(3, 0))
            parts.append(_encode_varint(self.act_dim))
        return b"".join(parts)


class _ResetRequest:
    """Manual serialization for ResetRequest."""

    def __init__(self, seed: int = 0, has_seed: bool = False) -> None:
        self.seed = seed
        self.has_seed = has_seed

    def SerializeToString(self) -> bytes:
        parts: list[bytes] = []
        if self.seed:
            parts.append(_encode_tag(1, 0))
            parts.append(_encode_varint(self.seed))
        if self.has_seed:
            parts.append(_encode_tag(2, 0))
            parts.append(_encode_varint(1))
        return b"".join(parts)


class _StepResponse:
    """Manual deserialization for StepResponse."""

    def __init__(self) -> None:
        self.obs: list[float] = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []
        self.obs_dim: int = 0
        self.num_envs: int = 0

    @staticmethod
    def FromString(data: bytes) -> _StepResponse:
        resp = _StepResponse()
        pos = 0
        while pos < len(data):
            field_num, wire_type, pos = _decode_tag(data, pos)
            if field_num == 1 and wire_type == 2:  # packed floats
                length, pos = _decode_varint(data, pos)
                end = pos + length
                while pos < end:
                    val = struct.unpack_from("<f", data, pos)[0]
                    resp.obs.append(val)
                    pos += 4
            elif field_num == 2 and wire_type == 2:  # packed doubles
                length, pos = _decode_varint(data, pos)
                end = pos + length
                while pos < end:
                    val = struct.unpack_from("<d", data, pos)[0]
                    resp.rewards.append(val)
                    pos += 8
            elif field_num == 3 and wire_type == 2:  # packed bools
                length, pos = _decode_varint(data, pos)
                end = pos + length
                while pos < end:
                    val, pos = _decode_varint(data, pos)
                    resp.terminated.append(bool(val))
            elif field_num == 4 and wire_type == 2:  # packed bools
                length, pos = _decode_varint(data, pos)
                end = pos + length
                while pos < end:
                    val, pos = _decode_varint(data, pos)
                    resp.truncated.append(bool(val))
            elif field_num == 5 and wire_type == 0:
                resp.obs_dim, pos = _decode_varint(data, pos)
            elif field_num == 6 and wire_type == 0:
                resp.num_envs, pos = _decode_varint(data, pos)
            else:
                pos = _skip_field(data, pos, wire_type)
        return resp


class _ResetResponse:
    """Manual deserialization for ResetResponse."""

    def __init__(self) -> None:
        self.obs: list[float] = []
        self.obs_dim: int = 0
        self.num_envs: int = 0

    @staticmethod
    def FromString(data: bytes) -> _ResetResponse:
        resp = _ResetResponse()
        pos = 0
        while pos < len(data):
            field_num, wire_type, pos = _decode_tag(data, pos)
            if field_num == 1 and wire_type == 2:
                length, pos = _decode_varint(data, pos)
                end = pos + length
                while pos < end:
                    val = struct.unpack_from("<f", data, pos)[0]
                    resp.obs.append(val)
                    pos += 4
            elif field_num == 2 and wire_type == 0:
                resp.obs_dim, pos = _decode_varint(data, pos)
            elif field_num == 3 and wire_type == 0:
                resp.num_envs, pos = _decode_varint(data, pos)
            else:
                pos = _skip_field(data, pos, wire_type)
        return resp


class _SpacesResponse:
    """Manual deserialization for SpacesResponse."""

    def __init__(self) -> None:
        self.action_space_json: str = "{}"
        self.obs_space_json: str = "{}"
        self.num_envs: int = 0

    @staticmethod
    def FromString(data: bytes) -> _SpacesResponse:
        resp = _SpacesResponse()
        pos = 0
        while pos < len(data):
            field_num, wire_type, pos = _decode_tag(data, pos)
            if field_num == 1 and wire_type == 2:
                length, pos = _decode_varint(data, pos)
                resp.action_space_json = data[pos:pos + length].decode("utf-8")
                pos += length
            elif field_num == 2 and wire_type == 2:
                length, pos = _decode_varint(data, pos)
                resp.obs_space_json = data[pos:pos + length].decode("utf-8")
                pos += length
            elif field_num == 3 and wire_type == 0:
                resp.num_envs, pos = _decode_varint(data, pos)
            else:
                pos = _skip_field(data, pos, wire_type)
        return resp


# ---------------------------------------------------------------------------
# Protobuf wire format helpers
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint starting at pos, return (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a protobuf field tag."""
    return _encode_varint((field_number << 3) | wire_type)


def _decode_tag(data: bytes, pos: int) -> tuple[int, int, int]:
    """Decode a protobuf field tag, return (field_number, wire_type, new_pos)."""
    tag, pos = _decode_varint(data, pos)
    return tag >> 3, tag & 0x07, pos


def _skip_field(data: bytes, pos: int, wire_type: int) -> int:
    """Skip an unknown field in the wire format."""
    if wire_type == 0:  # varint
        _, pos = _decode_varint(data, pos)
    elif wire_type == 1:  # 64-bit
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, pos = _decode_varint(data, pos)
        pos += length
    elif wire_type == 5:  # 32-bit
        pos += 4
    else:
        raise ValueError(f"Unknown wire type: {wire_type}")
    return pos
