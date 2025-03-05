# Copyright 2025 Lorenzo Sani & Alexandru-Andrei Iacob
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation Client Class for Flower framework."""

from collections.abc import Callable
import copy
from typing import Any
import pickle
from pathlib import Path
from logging import INFO
import numpy as np
from flwr.common.logger import log
from flwr.common.typing import Scalar, NDArrays
from flwr.client import NumPyClient, Client
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)

from .utils import (
    SecAggStages,
    ShareKeysPacket,
    bytes_to_private_key,
    bytes_to_public_key,
    create_shares,
    decrypt,
    encrypt,
    factor_weights_combine,
    generate_key_pairs,
    generate_shared_key,
    load_content,
    private_key_to_bytes,
    pseudo_rand_gen,
    public_key_to_bytes,
    quantize,
    rand_bytes,
    save_content,
    share_keys_plaintext_concat,
    share_keys_plaintext_separate,
    weights_addition,
    weights_mod,
    weights_multiply,
    weights_subtraction,
)


class SecureAggregationClient(NumPyClient):
    """Flower client leveraging secure aggregation."""

    def __init__(self, cid: int, n_dim: int, n_samples: int, data_dir: Path) -> None:
        self.cid = cid
        self.cache_pth = data_dir / f"{cid!s}.pth"
        self.model_parameters = [np.zeros(n_dim)]
        self.n_samples = n_samples

        # SecAgg specific parameters
        self.sec_agg_id: int | None = None
        self.sec_id: int | None = None
        self.sample_num: int | None = None
        self.share_num: int | None = None
        self.threshold: int | None = None
        self.drop_flag: bool | None = None
        self.clipping_range: int | None = None
        self.target_range: int | None = None
        self.mod_range: int | None = None
        self.b_share_dict: dict[int, bytes] | None = None
        self.sk1_share_dict: dict[int, bytes] | None = None
        self.shared_key_2_dict: dict[int, bytes] | None = None
        self.sk1: EllipticCurvePrivateKey | None = None
        self.pk1: EllipticCurvePublicKey | None = None
        self.sk2: EllipticCurvePrivateKey | None = None
        self.pk2: EllipticCurvePublicKey | None = None
        self.sk1_bytes: bytes | None = None
        self.pk1_bytes: bytes | None = None
        self.sk2_bytes: bytes | None = None
        self.pk2_bytes: bytes | None = None
        self.share_keys_dict: dict[int, tuple[bytes, bytes]] | None = None
        self.public_keys_dict: dict[int, tuple[bytes, bytes]] | None = None
        self.b: bytes | None = None

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Receive and a model on the local client data.

        It uses the instruction passed through the config dict.
        Note: no actual training is performed here.

        Args:
            net (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the training parameters

        Returns
        -------
            tuple[NDArrays, int, dict]: Returns the updated model, the size of the local
                dataset and other metrics
        """
        self.reload()
        stage = config.pop("stage")
        ret: Any = None
        ndarrays_object: NDArrays = []

        if stage == SecAggStages.STAGE_0:
            ret = setup_param(self, config)
        elif stage == SecAggStages.STAGE_1:
            ret = share_keys(self, load_content(config))
        elif stage == SecAggStages.STAGE_2:
            packet_lst, _fit_ins = load_content(config)
            # log(INFO, f'Client {self.sec_agg_id}: \n' + str(packet_lst))
            ndarrays_object = ask_vectors(self, packet_lst)
        elif stage == SecAggStages.STAGE_3:
            actives, dropouts = load_content(config)
            ret = unmask_vectors(self, actives, dropouts)

        self.cache()
        return ndarrays_object, 0, save_content(ret, {})

    def get_vars(self) -> dict[str, Any]:
        """Return all variables of the class as a dictionary."""
        return vars(self)

    def cache(self) -> None:
        """Cache the variables of the class to a file."""
        # Removing the key objects from the dictionary to prevent weird pickling errors
        current_vars = self.get_vars()
        current_vars.pop("sk1", None)
        current_vars.pop("pk1", None)
        current_vars.pop("sk2", None)
        current_vars.pop("pk2", None)
        with open(self.cache_pth, "wb") as f:
            pickle.dump(copy.deepcopy(current_vars), f)
        assert self.sk1_bytes is not None, "Private key 1 bytes is None"
        assert self.pk1_bytes is not None, "Public key 1 bytes is None"
        assert self.sk2_bytes is not None, "Private key 2 bytes is None"
        assert self.pk2_bytes is not None, "Public key 2 bytes is None"
        # Set back the keys from the bytes representation
        self.sk1 = bytes_to_private_key(self.sk1_bytes)
        self.pk1 = bytes_to_public_key(self.pk1_bytes)
        self.sk2 = bytes_to_private_key(self.sk2_bytes)
        self.pk2 = bytes_to_public_key(self.pk2_bytes)

    def reload(self) -> None:
        """Reload the variables of the class from a file."""
        if self.cache_pth.exists():
            log(INFO, "CID %s reloading from %s", self.cid, self.cache_pth)
            with open(self.cache_pth, "rb") as f:
                self.__dict__.update(pickle.load(f))
            assert self.sk1_bytes is not None, "Private key 1 bytes is None"
            assert self.pk1_bytes is not None, "Public key 1 bytes is None"
            assert self.sk2_bytes is not None, "Private key 2 bytes is None"
            assert self.pk2_bytes is not None, "Public key 2 bytes is None"
            # Set back the keys from the bytes representation
            self.sk1 = bytes_to_private_key(self.sk1_bytes)
            self.pk1 = bytes_to_public_key(self.pk1_bytes)
            self.sk2 = bytes_to_private_key(self.sk2_bytes)
            self.pk2 = bytes_to_public_key(self.pk2_bytes)


def get_sec_agg_client_generator(
    n_dim: int, n_samples: int, data_dir: Path
) -> Callable[[str], Client]:
    """Return a generator for SecureAggregationClient instances."""

    def client_fn(cid: str) -> Client:
        """Return a new SecureAggregationClient instance."""
        return SecureAggregationClient(int(cid), n_dim, n_samples, data_dir).to_client()

    return client_fn


def setup_param(
    client: SecureAggregationClient, setup_param_dict: dict[str, Scalar]
) -> tuple[bytes, bytes]:
    """Assign parameter values to object fields."""
    # Assigning parameter values to object fields
    sec_agg_param_dict = setup_param_dict
    client.sample_num = int(sec_agg_param_dict["share_num"])
    client.sec_id = int(sec_agg_param_dict["id"])
    client.sec_agg_id = int(sec_agg_param_dict["id"])
    log(INFO, "Client %s: starting stage 0...", client.sec_agg_id)

    client.share_num = int(sec_agg_param_dict["share_num"])
    client.threshold = int(sec_agg_param_dict["threshold"])
    client.drop_flag = bool(sec_agg_param_dict["drop_flag"])
    client.clipping_range = 3
    client.target_range = 1 << 16
    client.mod_range = 1 << 24

    # The key is the sec_agg_id of another client (int), while the value is the secret
    # share we possess that contributes to the client's secret (bytes)
    client.b_share_dict = {}
    client.sk1_share_dict = {}
    client.shared_key_2_dict = {}
    return ask_keys(client)


def ask_keys(client: SecureAggregationClient) -> tuple[bytes, bytes]:
    """Create and upload public keys for secure aggregation."""
    client.sk1, client.pk1 = generate_key_pairs()
    client.sk2, client.pk2 = generate_key_pairs()

    client.sk1_bytes, client.pk1_bytes = private_key_to_bytes(
        client.sk1
    ), public_key_to_bytes(client.pk1)
    client.sk2_bytes, client.pk2_bytes = private_key_to_bytes(
        client.sk2
    ), public_key_to_bytes(client.pk2)
    log(
        INFO,
        "Client %s: stage 0 completes. uploading public keys...",
        client.sec_agg_id,
    )
    return client.pk1_bytes, client.pk2_bytes


def share_keys(
    client: SecureAggregationClient, share_keys_dict: dict[int, tuple[bytes, bytes]]
) -> list[ShareKeysPacket]:
    """Implement the first stage of secure aggregation on the client side."""
    log(INFO, "Client %s: starting stage 1...", client.sec_agg_id)
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    client.public_keys_dict = share_keys_dict
    assert client.public_keys_dict is not None, "Public keys dict is None"
    assert client.threshold is not None, "Threshold is None"
    # check size is larger than threshold
    if len(client.public_keys_dict) < client.threshold:
        raise Exception(  # noqa: TRY002
            "Available neighbours number smaller than threshold"
        )

    # check if all public keys received are unique
    pk_list: list[bytes] = []
    for i in client.public_keys_dict.values():
        pk_list.extend([i[0], i[1]])
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")  # noqa: TRY002

    assert client.sec_agg_id is not None, "Secure aggregation ID is None"
    # sanity check that own public keys are correct in dict
    if (
        client.public_keys_dict[client.sec_agg_id][0] != client.pk1_bytes
        or client.public_keys_dict[client.sec_agg_id][1] != client.pk2_bytes
    ):
        raise Exception(  # noqa: TRY002
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate private mask seed
    client.b = rand_bytes(32)

    assert client.share_num is not None, "Share number is None"
    assert client.sk1_bytes is not None, "Private key 1 bytes is None"
    # Create shares
    b_shares = create_shares(client.b, client.threshold, client.share_num)
    sk1_shares = create_shares(client.sk1_bytes, client.threshold, client.share_num)

    share_keys_res_list = []

    assert client.b_share_dict is not None, "B share dict is None"
    assert client.sk1_share_dict is not None, "SK1 share dict is None"
    for idx, p in enumerate(client.public_keys_dict.items()):
        client_sec_agg_id, client_public_keys = p
        if client_sec_agg_id == client.sec_agg_id:
            client.b_share_dict[client.sec_agg_id] = b_shares[idx]
            client.sk1_share_dict[client.sec_agg_id] = sk1_shares[idx]
        else:
            assert client.sk2_bytes is not None, "Private key 2 bytes is None"
            shared_key = generate_shared_key(
                bytes_to_private_key(client.sk2_bytes),
                bytes_to_public_key(client_public_keys[1]),
            )
            assert client.shared_key_2_dict is not None, "Shared key 2 dict is None"
            client.shared_key_2_dict[client_sec_agg_id] = shared_key
            plaintext = share_keys_plaintext_concat(
                client.sec_agg_id, client_sec_agg_id, b_shares[idx], sk1_shares[idx]
            )
            ciphertext = encrypt(shared_key, plaintext)
            share_keys_packet = ShareKeysPacket(
                source=client.sec_agg_id,
                destination=client_sec_agg_id,
                ciphertext=ciphertext,
            )
            share_keys_res_list.append(share_keys_packet)

    log(
        INFO, "Client %s: stage 1 completes. uploading key shares...", client.sec_agg_id
    )
    return share_keys_res_list


def ask_vectors(client: SecureAggregationClient, packet_list: Any) -> NDArrays:
    """Implement the second stage of secure aggregation onm the client side."""
    log(INFO, "Client %s: starting stage 2...", client.sec_agg_id)
    # Receive shares and fit model
    available_clients: list[int] = []

    assert client.threshold is not None, "Threshold is None"
    if len(packet_list) + 1 < client.threshold:
        raise Exception(  # noqa: TRY002
            "Available neighbours number smaller than threshold"
        )

    # decode all packets and verify all packets are valid. Save shares received
    for packet in packet_list:
        source = packet.source
        available_clients.append(source)
        destination = packet.destination
        ciphertext = packet.ciphertext
        if destination != client.sec_agg_id:
            raise Exception(  # noqa: TRY002
                "Received packet meant for another user. Not supposed to happen"
            )
        assert client.shared_key_2_dict is not None, "Shared key 2 dict is None"
        shared_key = client.shared_key_2_dict[source]
        plaintext = decrypt(shared_key, ciphertext)
        try:
            (
                plaintext_source,
                plaintext_destination,
                plaintext_b_share,
                plaintext_sk1_share,
            ) = share_keys_plaintext_separate(plaintext)
        except Exception as e:
            raise Exception(  # noqa: TRY002
                "Decryption of ciphertext failed. Not supposed to happen"
            ) from e
        if plaintext_source != source:
            raise Exception(  # noqa: TRY002
                "Received packet source is different from intended source."
                "Not supposed to happen"
            )
        if plaintext_destination != destination:
            raise Exception(  # noqa: TRY002
                "Received packet destination is different from intended destination."
                "Not supposed to happen"
            )
        assert client.b_share_dict is not None, "B share dict is None"
        assert client.sk1_share_dict is not None, "SK1 share dict is None"
        client.b_share_dict[source] = plaintext_b_share
        client.sk1_share_dict[source] = plaintext_sk1_share

    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    """
    fit_res = client.client.fit(fit_ins)
    parameters = fit_res.parameters
    weights = parameters_to_weights(parameters)
    weights_factor = fit_res.num_examples
    """

    if client.drop_flag:
        # log(ERROR, "Force dropout due to testing!!")
        raise Exception("Force dropout due to testing")  # noqa: TRY002
    # weights = model_parameters
    weights = client.model_parameters
    # weights_factor = weight
    weights_factor = client.n_samples

    assert client.clipping_range is not None, "Clipping range is None"
    assert client.target_range is not None, "Target range is None"
    # Quantize weight update vector
    quantized_weights = quantize(weights, client.clipping_range, client.target_range)

    quantized_weights = weights_multiply(quantized_weights, weights_factor)
    quantized_weights = factor_weights_combine(weights_factor, quantized_weights)

    dimensions_list: list[tuple] = [a.shape for a in quantized_weights]

    # add private mask
    assert client.b is not None, "B is None"

    assert client.mod_range is not None, "Mod range is None"
    private_mask = pseudo_rand_gen(client.b, client.mod_range, dimensions_list)
    quantized_weights = weights_addition(quantized_weights, private_mask)

    for client_id in available_clients:
        assert client.sk1_bytes is not None, "Private key 1 bytes is None"
        assert client.public_keys_dict is not None, "Public keys dict is None"
        # add pairwise mask
        shared_key = generate_shared_key(
            bytes_to_private_key(client.sk1_bytes),
            bytes_to_public_key(client.public_keys_dict[client_id][0]),
        )
        # print('shared key length: %d' % len(shared_key))
        pairwise_mask = pseudo_rand_gen(shared_key, client.mod_range, dimensions_list)
        assert client.sec_agg_id is not None, "Secure aggregation ID is None"
        if client.sec_agg_id > client_id:
            quantized_weights = weights_addition(quantized_weights, pairwise_mask)
        else:
            quantized_weights = weights_subtraction(quantized_weights, pairwise_mask)

    # Take mod of final weight update vector and return to server
    quantized_weights = weights_mod(quantized_weights, client.mod_range)
    # return ndarrays_to_parameters(quantized_weights)
    log(
        INFO,
        "Client %s: stage 2 completes. uploading masked weights...",
        client.sec_agg_id,
    )
    return quantized_weights


def unmask_vectors(
    client: SecureAggregationClient, available_clients: Any, dropout_clients: Any
) -> dict[int, bytes]:
    """Send private mask seed share for every avaliable client (including itclient).

    Send first private key share for building pairwise mask for every dropped client.
    """
    assert client.threshold is not None, "Threshold is None"
    assert client.sk1_share_dict is not None, "Private keys 1 share dict is None"
    assert client.b_share_dict is not None, "B share dict is None"
    if len(available_clients) < client.threshold:
        raise Exception(  # noqa: TRY002
            "Available neighbours number smaller than threshold"
        )
    share_dict: dict[int, bytes] = {}
    for idx in available_clients:
        share_dict[idx] = client.b_share_dict[idx]
    for idx in dropout_clients:
        share_dict[idx] = client.sk1_share_dict[idx]
    return share_dict
