from core.models.pinn import PINN

import copy


class Node:
    def __init__(self, pinn: PINN | None, total_epoch: int, id: int, possibles_params: list[tuple[str, str]], current_steps: int, factor: str):
        self.pinn = copy.deepcopy(pinn) if pinn is not None else None
        self.total_epoch = total_epoch
        self.id = id
        self.possibles_params = copy.deepcopy(possibles_params)
        self.current_steps = current_steps
        self.factor = factor

    def __str__(self):
        return f"Node(id={self.id}, pinn={self.pinn is not None}, total_epoch={self.total_epoch}, current_steps={self.current_steps}, factor={self.factor})"


class Edge:
    def __init__(self, parent: Node, child: Node, factor: str, optimizer: str, epoch: int):
        self.parent = parent
        self.child = child
        self.factor = factor
        self.optimizer = optimizer
        self.epoch = epoch

        self.parent_id = parent.id
        self.child_id = child.id

    def __str__(self):
        return f"Edge(parent_id={self.parent_id}, child_id={self.child_id}, factor={self.factor}, optimizer={self.optimizer}, epoch={self.epoch})"



