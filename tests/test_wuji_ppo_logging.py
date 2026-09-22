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


def test_episode_window_weights_episodes_and_survives_generation_reset():
    import torch
    run = Run()
    window = m.CompletedEpisodeWindow(window_steps=3, global_envs=3, log=run.log)
    def step(rewards, done):
        window.observe(torch.tensor(rewards), torch.tensor(done), torch.zeros(3, dtype=torch.bool))
    step([1., 2., 3.], [True, False, False])
    step([9., 4., 5.], [False, True, True])
    # Complete returns: 1, 6, 8. The unfinished return of 9 must not carry over.
    window.finish_generation()
    step([10., 20., 30.], [True, False, False])
    record = run.logs[0]
    prefix = 'PPO/Completed episodes / '
    assert record[prefix + 'Return mean (window)'] == 25 / 4
    assert record[prefix + 'Length mean (window)'] == 6 / 4
    assert record[prefix + 'Count (window)'] == 4
    assert record[prefix + 'Interrupted episodes (total)'] == 1
    assert record['PPO/train_steps'] == 3
    assert record['PPO/train_transitions'] == 9
    window.finish_generation()
    window.flush()
    assert run.logs[-1][prefix + 'Interrupted episodes (total)'] == 3
    assert prefix + 'Count (window)' not in run.logs[-1]


def test_episode_window_counts_all_environments_and_includes_terminal_reward():
    import torch
    run = Run()
    window = m.CompletedEpisodeWindow(window_steps=2, global_envs=201, log=run.log)
    first = torch.arange(201, dtype=torch.float32).reshape(-1, 1)
    flags = torch.zeros_like(first, dtype=torch.bool)
    window.observe(first, flags, flags)
    window.observe(torch.ones_like(first), flags, ~flags)  # timeouts also complete episodes
    record = run.logs[0]
    assert record['PPO/Completed episodes / Count (window)'] == 201
    assert record['PPO/Completed episodes / Return mean (window)'] == 101
    assert record['PPO/Completed episodes / Length mean (window)'] == 2


def test_empty_window_does_not_invent_zero_return_and_partial_window_is_flushed():
    import torch
    run = Run()
    window = m.CompletedEpisodeWindow(window_steps=2, global_envs=1, log=run.log)
    def step(done):
        window.observe(torch.tensor([2.]), torch.tensor([done]), torch.tensor([False]))
    step(False)
    step(False)
    assert run.logs[0]['PPO/Completed episodes / Count (window)'] == 0
    assert 'PPO/Completed episodes / Return mean (window)' not in run.logs[0]
    step(True)
    window.flush()
    assert run.logs[-1]['PPO/Completed episodes / Window steps'] == 1
    assert run.logs[-1]['PPO/Completed episodes / Return mean (window)'] == 6
    assert run.logs[-1]['PPO/Completed episodes / Length mean (window)'] == 3


def test_record_hook_preserves_training_inputs_and_observes_before_shaping():
    import torch
    run = Run()
    window = m.CompletedEpisodeWindow(window_steps=1, global_envs=1, log=run.log)
    rewards = torch.tensor([[7.]])
    def original_record(**kwargs):
        assert kwargs['rewards'] is rewards
        assert rewards.item() == 7
        kwargs['rewards'].add_(100)  # mimic a later training-only bootstrap
        return 'unchanged-return'
    agent = SimpleNamespace(record_transition=original_record)
    m.track_completed_episodes(agent, window)
    assert agent.record_transition(rewards=rewards, terminated=torch.tensor([[True]]),
                                   truncated=torch.tensor([[False]])) == 'unchanged-return'
    assert run.logs[0]['PPO/Completed episodes / Return mean (window)'] == 7


def test_distributed_window_reduces_sums_and_counts_before_computing_mean(monkeypatch):
    import torch
    run = Run()
    def reduce(totals):
        # A second rank contributed two episodes, with returns 10 and 20.
        totals.add_(torch.tensor([30., 2., 2., 0.]))
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    window = m.CompletedEpisodeWindow(window_steps=1, global_envs=3, log=run.log, distributed=True)
    window.observe(torch.tensor([3.]), torch.tensor([True]), torch.tensor([False]))
    assert run.logs[0]['PPO/Completed episodes / Return mean (window)'] == 11
    assert run.logs[0]['PPO/Completed episodes / Count (window)'] == 3
