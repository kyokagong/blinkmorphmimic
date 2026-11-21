"""Preview actor capture and shared physics graph."""
import torch
from morphmimic.env.control_graph import PhysicsGraph


class ActorGraph:
    def __init__(self, actor, device):
        self.actor, self.device = actor, device
        self.graph = self.inputs = self.output = None

    def __call__(self, *inputs):
        if self.graph is None:
            self.inputs = tuple(value.detach().clone() for value in inputs)
            with torch.cuda.device(self.device):
                self.actor(*self.inputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.actor(*self.inputs)
                graph.replay()
                torch.cuda.synchronize(self.device)
            self.graph, self.output = graph, output
        else:
            for target, value in zip(self.inputs, inputs, strict=True):
                if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                    raise ValueError('Actor graph input signature changed')
                target.copy_(value)
            self.graph.replay()
        return self.output
