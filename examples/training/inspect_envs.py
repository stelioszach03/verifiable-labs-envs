"""One-shot reconnaissance of env score() schemas for the multi-env design."""
from verifiable_labs_envs import load_environment
from verifiable_labs_envs.solvers.llm_solver import get_adapter

ENVS = [
    "sparse-fourier-recovery",
    "phase-retrieval",
    "super-resolution-div2k-x4",
    "mri-knee-reconstruction",
]

for eid in ENVS:
    print(f"=== {eid} ===")
    try:
        env = load_environment(eid, calibration_quantile=2.0)
        adapter = get_adapter(eid)
        inst = env.generate_instance(seed=0)
        print(f"  env: {type(env).__name__}")
        print(f"  adapter: {type(adapter).__name__}")
        print(f"  instance: {type(inst).__name__}")
        if hasattr(inst, "__dataclass_fields__"):
            print(f"  instance fields: {list(inst.__dataclass_fields__.keys())}")
        if hasattr(inst, "as_inputs"):
            inputs = inst.as_inputs()
            print(f"  as_inputs keys: {list(inputs.keys())}")
        if hasattr(env, "hyperparams"):
            print(f"  hyperparams: {env.hyperparams}")
        # score
        score = env.run_baseline(seed=0)
        print(f"  score keys: {list(score.keys())}")
        if "components" in score:
            print(f"  score.components: {list(score['components'].keys())}")
        if "meta" in score:
            print(f"  score.meta: {list(score['meta'].keys())}")
        print(f"  baseline reward: {score.get('reward', 'N/A')}")
    except Exception as e:
        import traceback
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
    print()
