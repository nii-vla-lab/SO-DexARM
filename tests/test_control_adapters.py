from types import SimpleNamespace

from sodexarm.control._adapters import adapt_control_io, make_record_loop_adapter


class FakeRobot:
    action_features = {"arm.pos": float, "hand.pos": float}

    def __init__(self):
        self.observations = 0

    def get_observation(self):
        self.observations += 1
        return {"arm.pos": 12.0, "hand.pos": 34.0}

    def should_abort_episode(self):
        return False


class FakeTeleop:
    def __init__(self):
        self.feedback = None
        self.began = 0
        self.ended = 0

    def get_action(self):
        return {"hand.pos": 50.0}

    def send_feedback(self, observation):
        self.feedback = observation

    def is_control_ready(self):
        return True

    def prepare_control_session(self, robot):
        self.robot = robot

    def begin_episode(self, robot, events=None):
        self.began += 1

    def end_episode(self, robot):
        self.ended += 1

    def should_abort_episode(self):
        return False


def test_io_adapter_sends_feedback_and_completes_hand_only_action() -> None:
    robot = FakeRobot()
    teleop = FakeTeleop()
    with adapt_control_io(teleop, robot, fps=30):
        observation = robot.get_observation()
        action = teleop.get_action()

    assert teleop.feedback == observation
    assert action == {"arm.pos": 12.0, "hand.pos": 50.0}


def test_record_adapter_keeps_upstream_loop_and_adds_lifecycle() -> None:
    robot = FakeRobot()
    teleop = FakeTeleop()
    events = {"rerecord_episode": False}
    dataset = SimpleNamespace(
        num_episodes=0,
        root=".",
        save_episode=lambda: None,
    )

    def upstream_loop(**kwargs):
        kwargs["robot"].get_observation()
        kwargs["teleop"].get_action()
        return "upstream-result"

    result = make_record_loop_adapter(upstream_loop)(
        robot=robot,
        teleop=teleop,
        events=events,
        dataset=dataset,
        fps=30,
    )

    assert result == "upstream-result"
    assert teleop.began == 1
    assert teleop.ended == 1
    assert teleop.robot is robot
