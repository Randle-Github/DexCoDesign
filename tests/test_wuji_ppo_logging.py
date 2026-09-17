import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wuji_ppo_logging', ROOT / 'temp/hocap_mano_replay/scripts/wuji_ppo_logging.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Run:
    def __init__(self):
        self.logs = []
        self.definitions = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, metrics):
        self.logs.append(metrics)


def test_logging_preserves_writer_and_continues_across_generations():
    run = Run()
    calls = []
    for generation in range(2):
        data = {'Reward / Total reward (mean)': [10, 20],
                'Reward / Instantaneous reward (max)': [1, 3],
                'Reward / Instantaneous reward (min)': [-2, -1],
                'Loss / Value loss': [4, 6], 'empty': []}
        def original_write(*, timestep, timesteps):
            calls.append((timestep, timesteps))
            data.clear()
        agent = SimpleNamespace(tracking_data=data, write_tracking_data=original_write)
        m.enable_ppo_logging(agent, run, generation=generation, steps_per_generation=1024, global_envs=4096)
        agent.write_tracking_data(timestep=16, timesteps=1024)
        assert not data
    assert len(calls) == 2
    assert [v['PPO/train_steps'] for v in run.logs] == [16, 1040]
    assert run.logs[1]['PPO/train_transitions'] == 1040 * 4096
    assert run.logs[0]['PPO/Reward / Episode return (mean)'] == 15
    assert run.logs[0]['PPO/Reward / Instantaneous reward (max)'] == 3
    assert run.logs[0]['PPO/Reward / Instantaneous reward (min)'] == -2
    assert run.logs[0]['PPO/Loss / Value loss'] == 5
    assert 'PPO/empty' not in run.logs[0]
