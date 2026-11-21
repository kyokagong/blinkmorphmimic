"""Fixed-shape physics capture with host-side safety checks between steps."""
import torch
from contextlib import nullcontext


class PhysicsGraph:
    def __init__(self, env, stream=None):
        self.env = env
        self.stream = stream if stream is not None else getattr(env, 'tensor_step_stream', None)
        self.reset()

    def reset(self):
        if getattr(self, 'graph', None) is not None and self.stream is not None:
            self.stream.synchronize()
        self.graph = self.action = self.output = self.warp_graph = self.signature = None

    def warmup(self, action):
        if self.graph is not None:
            raise ValueError('Warmup must precede graph capture')
        if not bool(torch.isfinite(action).all()):
            raise ValueError('Nonfinite warmup action')
        self.env.validate_tensor_step()
        physics = self.env.physics_diagnostics.report()
        if physics['overflow_bits']:
            raise RuntimeError(f'Native physics overflow before graph warmup: {physics}')
        if self.stream is None:
            self.env.step_no_reset_tensor(action)
        else:
            import warp as wp
            current = torch.cuda.current_stream(self.env.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream), wp.ScopedStream(wp.stream_from_torch(self.stream)):
                self.env.step_no_reset_tensor(action)
            current.wait_stream(self.stream)
        physics = self.env.physics_diagnostics.report()
        if physics['overflow_bits']:
            raise RuntimeError(f'Native physics overflow during graph warmup: {physics}')

    def step(self, action):
        self.env.validate_tensor_step()
        signature_fn = getattr(self.env, '_sim_array_signature', None)
        if self.graph is not None and signature_fn is not None and signature_fn() != self.signature:
            raise RuntimeError('Simulator arrays changed; recreate the environment and captured graphs')
        if self.graph is None:
            self.action = action.detach().clone()
        else:
            if (self.action.shape != action.shape or self.action.dtype != action.dtype
                    or self.action.device != action.device):
                raise ValueError('Physics graph action signature changed')
            self.action.copy_(action)
        warp_scope = nullcontext()
        if self.stream is not None:
            import warp as wp
            warp_stream = wp.stream_from_torch(self.stream)
            warp_scope = wp.ScopedStream(warp_stream, sync_enter=False, sync_exit=False)
            self.stream.wait_stream(torch.cuda.current_stream(self.env.device))
        with warp_scope:
            if self.graph is None:
                with torch.cuda.device(self.env.device):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, capture_error_mode='thread_local', **({'stream': self.stream} if self.stream is not None else {})):
                        if self.stream is None:
                            output = self.env.step_no_reset_tensor(self.action)
                        else:
                            wp.capture_begin(stream=warp_stream, external=True)
                            try:
                                output = self.env.step_no_reset_tensor(self.action)
                            finally:
                                self.warp_graph = wp.capture_end(stream=warp_stream)
                self.graph, self.output = graph, output
                self.signature = signature_fn() if signature_fn is not None else None
                with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
                    graph.replay()
            else:
                with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
                    self.graph.replay()
                self.env._sim_step_counter += self.env.cfg.decimation
                self.env.common_step_counter += 1
        if self.stream is not None:
            torch.cuda.current_stream(self.env.device).wait_stream(self.stream)
        return self.output


