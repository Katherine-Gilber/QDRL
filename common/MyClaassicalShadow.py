import pennylane as qml
import numpy as np
import torch
from string import ascii_letters as ABC
class MyClassicalShadow(qml.ClassicalShadow):
    def __init__(self,bits, recipes,wire_map=None):
        self.bits = bits
        self.recipes = recipes

        # the wires corresponding to the columns of bitstrings
        if wire_map is None:
            self.wire_map = list(range(bits.shape[1]))
        else:
            self.wire_map = wire_map

        if bits.shape != recipes.shape:
            raise ValueError(
                f"Bits and recipes but have the same shape, got {bits.shape} and {recipes.shape}."
            )

        if bits.shape[1] != len(self.wire_map):
            raise ValueError(
                f"The 1st axis of bits must have the same size as wire_map, got {bits.shape[1]} and {len(self.wire_map)}."
            )

        self.observables = [
            qml.matrix(qml.PauliX(0)), #0
            qml.matrix(qml.PauliY(0)), #1
            qml.matrix(qml.PauliZ(0)), #2
        ]

    def local_snapshots(self,fake_index, subsystem):
        #每行选不同
        # pick_system=[]
        # for _ in range(len(self.bits)):
        #     pick_system.append(np.random.choice(
        #             np.arange(len(fake_index),dtype=np.int64), size=subsystem, replace=False
        #         ))
        # pick_system = qml.math.convert_like(pick_system, self.bits)
        # # print(pick_system)
        # bits = torch.gather(self.bits,1,pick_system).cpu().numpy()
        # recipes = torch.gather(self.recipes,1, pick_system).cpu().numpy()
        # set_seed(42)
        #每行选相同
        pick_system = np.random.choice(
            np.arange(len(fake_index), dtype=np.int64), size=subsystem, replace=False)
        # print("pick_system:",pick_system)
        pick_system = qml.math.convert_like(pick_system, self.bits)
        bits = self.bits.index_select(1, pick_system)
        recipes = self.recipes.index_select(1, pick_system)
        T, n = bits.shape
        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(self.observables):
            U[np.where(recipes.cpu().detach().numpy() == i)] = u
        state = (qml.math.cast((1 - 2 * bits[:, :, None, None].cpu().detach().numpy()), np.complex64) * U + np.eye(2)) / 2
        return 3 * state - np.eye(2)[None,None,:,:]
        # pick_system = torch.randperm(len(fake_index))[:subsystem].to(self.bits.device)
        # print("pick_system:", pick_system)
        #
        # # Assuming self.bits and self.recipes are already PyTorch tensors
        # bits = self.bits.index_select(1, pick_system)
        # recipes = self.recipes.index_select(1, pick_system)
        # T, n = bits.shape
        # U = torch.zeros((T, n, 2, 2), dtype=torch.complex64)
        #
        # # Convert numpy arrays in self.observables to PyTorch tensors
        # for i, u in enumerate(self.observables):
        #     mask = (recipes == i)
        #     U[mask] = u
        #
        # # Element-wise operations with broadcasting
        # state = ((1 - 2 * bits.unsqueeze(-1).unsqueeze(-1)) * U + torch.eye(2, dtype=torch.complex64)) / 2
        # result = 3 * state - torch.eye(2, dtype=torch.complex64).unsqueeze(0).unsqueeze(0)
        #
        # return result


    def global_snapshots(self,fake_index, subsystem=None):
        local_snapshot = self.local_snapshots(fake_index, subsystem)

        T, n = local_snapshot.shape[:2]

        transposed_snapshots = np.transpose(local_snapshot, axes=(1, 0, 2, 3))

        old_indices = [f"a{ABC[1 + 2 * i: 3 + 2 * i]}" for i in range(n)]
        new_indices = f"a{ABC[1:2 * n + 1:2]}{ABC[2:2 * n + 1:2]}"

        return np.reshape(
            np.einsum(f'{",".join(old_indices)}->{new_indices}', *transposed_snapshots),
            (T, 2 ** n, 2 ** n),
        )

    def median_of_means(self,arr, num_batches, axis=0):
        batch_size=(arr.shape[0]+num_batches-1)//num_batches
        # batch_size=int(torch.ceil(arr.shape[0] / num_batches))
        means=[torch.mean(arr[i * batch_size: (i + 1) * batch_size], 0) for i in range(num_batches)]
        # return means
        # print("in median_of_means:",means,means[0].requires_grad)
        return torch.mean(torch.stack(means), dim=axis)

    def median_of_means_(self,arr, num_batches, axis=0):
        batch_size = int(np.ceil(arr.shape[0] / num_batches))
        means = [
            qml.math.mean(arr[i * batch_size: (i + 1) * batch_size], 0) for i in range(num_batches)
        ]
        return np.median(means, axis=axis)

    def project_density_matrix_spectrum(self,rdm):
        evs = qml.math.eigvalsh(rdm)[::-1]  # order from largest to smallest
        d = len(rdm)
        a = 0.0
        for i in range(d - 1, -1, -1):
            if evs[i] + a / (i + 1) > 0:
                break
            a =a+ evs[i]

        lambdas = evs[: i + 1] + a / (i + 1)
        return lambdas[::-1]

    def entropy(self, fake_index, subsystem, alpha=2, k=1, base=None, atol=1e-5):
        global_snapshots = self.global_snapshots(fake_index, subsystem=subsystem)
        # rdm = self.median_of_means(global_snapshots, k, axis=0)
        rdm = self.median_of_means_(global_snapshots, k, axis=0)

        # Allow for different log base
        div = np.log(base) if base else 1

        evs_nonzero = self.project_density_matrix_spectrum(rdm)
        if alpha == 1:
            # Special case of von Neumann entropy
            return qml.math.entr(evs_nonzero) / div

        # General Renyi-alpha entropy
        return qml.math.log(qml.math.sum(evs_nonzero ** alpha)) / (1.0 - alpha) / div

    def expval(self, H,device, k=1):
        # if not isinstance(H, (list, tuple)):
        #     H = [H]

        coeffs_and_words = [self._convert_to_pauli_words(h) for h in H]
        expvals = self.pauli_expval(torch.tensor([word for cw in coeffs_and_words for _, word in cw]),device)

        # print("平均之前expvals:",expvals,expvals.requires_grad) #True
        expvals = self.median_of_means(expvals, k, axis=0)


        return expvals
        # expvals = expvals * np.array([coeff for cw in coeffs_and_words for coeff, _ in cw])
        # start = 0
        # results = []
        # for i in range(len(H)):
        #     results.append(np.sum(expvals[start: start + len(coeffs_and_words[i])]))
        #     start += len(coeffs_and_words[i])
        # return qml.math.squeeze(results)
    def pauli_expval(self,word,device):
        # T, n = recipes.shape
        # b = word.shape[0]
        #
        # bits = qml.math.cast(bits, np.int64)
        # recipes = qml.math.cast(recipes, np.int64)
        #
        # word = qml.math.convert_like(qml.math.cast_like(word, bits), bits)
        #
        # # -1 in the word indicates an identity observable on that qubit
        # id_mask = word == -1
        #
        # # determine snapshots and qubits that match the word
        # indices = qml.math.equal(
        #     qml.math.reshape(recipes, (T, 1, n)), qml.math.reshape(word, (1, b, n))
        # )
        # tmp=qml.math.tile(qml.math.reshape(id_mask, (1, b, n)), (T, 1, 1))
        # indices = torch.logical_or(indices, tmp)
        #
        # indices = qml.math.all(indices, axis=2) #满足公式H要求的一次测量
        #
        # # mask identity bits (set to 0)
        # bits = qml.math.where(id_mask, 0, qml.math.tile(qml.math.expand_dims(bits, 1), (1, b, 1)))
        #
        # bits = qml.math.sum(bits, axis=2) % 2
        #
        # # print(np.count_nonzero(np.logical_not(id_mask),axis=1))
        # expvals = qml.math.where(indices, 1 - 2 * bits, 0) * 3 ** torch.count_nonzero(
        #     torch.logical_not(id_mask),dim=1
        # )
        # return qml.math.cast(expvals, np.float64)
        T, n = self.recipes.shape
        b = word.shape[0]

        word = word.to(self.bits.device)

        # -1 in the word indicates an identity observable on that qubit
        id_mask = (word == -1)

        # determine snapshots and qubits that match the word
        indices = torch.eq(
            self.recipes.view(T, 1, n), word.view(1, b, n)
        )
        indices = torch.logical_or(indices, id_mask.view(1, b, n).expand(T, b, n))
        indices = indices.all(dim=2)

        # mask identity bits (set to 0)


        bits = torch.where(id_mask, torch.tensor(0, dtype=self.bits.dtype, device=self.bits.device),
                           torch.tile(torch.unsqueeze(self.bits, 1), (1, b, 1)))

        bits = bits.sum(dim=2) % 2

        # non_zero_count = torch.count_nonzero(~id_mask, dim=1)
        expvals = torch.where(indices, 1 - 2 * self.bits, torch.tensor(0, dtype=self.bits.dtype, device=bits.device)) * \
                  3 ** torch.count_nonzero(torch.logical_not(id_mask),dim=1)

        return expvals.type(torch.float64).to(device).requires_grad_(True)