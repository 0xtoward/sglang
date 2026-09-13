import unittest
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner import metadata_glue_graph
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestMetadataGlueGraph(CustomTestCase):
    def test_capture_scope_and_cached_replay(self):
        backend = Mock(attn_backend_list=None)
        metadata = backend.forward_metadata
        graph_runner = metadata_glue_graph.MetadataGlueGraph("cuda")
        batch = object()

        with patch.object(metadata_glue_graph.torch, "cuda") as cuda:
            for _ in range(graph_runner.NUM_WARMUP + 1):
                graph_runner.run(backend, batch, key=1)

            graph = cuda.CUDAGraph.return_value
            cuda.graph.assert_called_once_with(
                graph,
                stream=cuda.Stream.return_value,
                capture_error_mode="thread_local",
            )
            self.assertEqual(graph.replay.call_count, 1)
            self.assertEqual(
                backend.init_forward_metadata_out_graph.call_count,
                graph_runner.NUM_WARMUP + 1,
            )

            backend.forward_metadata = object()
            graph_runner.run(backend, batch, key=1)
            self.assertIs(backend.forward_metadata, metadata)
            self.assertEqual(graph.replay.call_count, 2)
            cuda.graph.assert_called_once()

    def test_capture_failure_retains_eager_fallback(self):
        backend = Mock(attn_backend_list=None)
        graph_runner = metadata_glue_graph.MetadataGlueGraph("cuda")

        with patch.object(metadata_glue_graph.torch, "cuda") as cuda:
            cuda.graph.return_value.__enter__.side_effect = RuntimeError(
                "unsafe operation in the capturing thread"
            )
            with self.assertLogs(metadata_glue_graph.logger, level="WARNING"):
                for _ in range(graph_runner.NUM_WARMUP + 1):
                    graph_runner.run(backend, object(), key=1)

            self.assertTrue(graph_runner.disabled)
            self.assertEqual(
                backend.init_forward_metadata_out_graph.call_count,
                graph_runner.NUM_WARMUP + 1,
            )
            cuda.CUDAGraph.return_value.replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
