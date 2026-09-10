# tools/

Two independent utilities, owned by different layers - grouped here
because both are "offline tooling" rather than part of the on-device
runtime.

- [`export/`](export/README.md) - ONNX -> TFLite conversion scripts. **[Layer 1]**
- [`benchmark_replay/`](benchmark_replay/README.md) - offline trajectory replay + drift plots. **[Layer 2, Section 9]** - see [`../docs/HANDOFF_benchmark_replay_tool.md`](../docs/HANDOFF_benchmark_replay_tool.md) for the full build brief.
