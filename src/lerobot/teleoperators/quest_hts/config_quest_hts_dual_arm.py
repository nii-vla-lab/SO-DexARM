#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("quest_hts_dual_arm")
@dataclass
class QuestHTSDualArmTeleoperatorConfig(TeleoperatorConfig):
    """DualArm Quest HTS teleoperator: right + left SO-101 + AmazingHand via Meta Quest hand tracking.

    Uses one TCP server to receive Quest HTS data for both hands.
    Each hand's wrist pose drives the corresponding arm via constrained planar IK.
    Each hand's landmarks drive the AmazingHand via piecewise calibrated mapping.

    Ports:
      Right arm:  /dev/ttyso101_amazinghand_r_arm
      Left arm:   /dev/ttyso101_amazinghand_l_arm
      Right hand: /dev/ttyso101_amazinghand_r_hand
      Left hand:  /dev/ttyso101_amazinghand_l_hand
    """

    # TCP server for Quest HTS
    host: str = "0.0.0.0"
    port: int = 8000

    # Calibration / startup files (created by lerobot-so-dexarm calibrate-hand / capture-startup)
    startup_file: Path = Path(".cache/so_dexarm/quest_hts_dual_arm_startup.yaml")
    hand_calib_file: Path = Path(".cache/so_dexarm/quest_hts_dual_arm_hand_calibration.yaml")

    # Arm calibration files (JSON created by `lerobot calibrate`)
    right_arm_calib_file: Path = Path(".cache/calibration/robots/so101_amazinghand_right/right.json")
    left_arm_calib_file: Path = Path(".cache/calibration/robots/so101_amazinghand_left/left.json")

    # Planar IK scale: human wrist delta (m, Quest WORLD frame) → robot EE delta (m).
    # 1.0 = TRUE 1:1 (EE moves exactly the distance the hand moved, within the workspace).
    # Position deltas are read in the shared Quest world frame and rotated into the robot
    # frame, so BOTH arms use the SAME sign (no mirroring): moving either hand toward the
    # robot extends that arm forward. Flip a single value if one arm tracks the wrong way.
    # Raise above 1.0 to amplify (smaller hand envelope); lower to attenuate.
    # Both arms share the sign when mounted in parallel; flip one value if that side tracks depth
    # backwards in a different mounting.
    planar_scale_x_right: float = -1.0
    planar_scale_y_right: float = 1.0
    planar_scale_x_left: float = -1.0  # world frame: same sign as right (depth is shared)
    planar_scale_y_left: float = 1.0

    # Control mode: arm-only | hand-only | arm-and-hand
    mode: str = "arm-and-hand"

    # Keep the elbow from fully straightening (singularity). Planar reach targets are capped to
    # this fraction of (l1 + l2). At full extension the 2-link IK locks: the arm "can't get up"
    # and ignores a pull-back (you extend then pull toward your body but the arm won't retract).
    # Staying < 1.0 always leaves a usable bend so the arm can retract. 0.95 ≈ 2-3° from straight.
    max_reach_fraction: float = 0.95

    # Safety / step limits
    # max_controlled_joint_step is a per-tick SAFETY cap, not a motion limiter. Set it high
    # enough that normal hand motion never hits it (otherwise the arm rate-limits = lag), but
    # low enough to bound a tracking glitch. 20 units ≈ 24° per tick.
    max_controlled_joint_step: float = 20.0  # max joint change per tick (calibrated units)
    max_controlled_joint_delta_from_start: float = 15.0
    max_hand_joint_delta_from_start: float = 100.0

    # Grip "squeeze" boost (over-close). Push the closing fingers farther toward
    # their fully-closed end than the calibrated fist pose, proportional to how much of a fist the
    # human is making. A blocking object then sees a larger commanded close → larger position
    # error → the servo drives harder against it (up to its torque limit) = a firmer squeeze.
    # 0.0 = OFF (raw mapper output, the calibrated fist).  0.3 ≈ push 30% of the open→fist span
    # past the fist pose at a full fist (clamped to the calibrated range).  Only acts while
    # closing — an open hand is untouched. Raise for a tighter grip; lower if fingers strain/buzz.
    grip_close_boost: float = 0.3

    # Thumb (finger1 = motor IDs 1-2) closing GAIN. The thumb's tip-to-MCP tracking feature has a
    # much smaller dynamic range than the other fingers (it barely shrinks when you make a fist —
    # the thumb folds ACROSS the palm rather than curling), so the mapped thumb under-bends. This
    # multiplies finger1's mapped closedness (0=open … 1=fist) so a partial thumb close bends
    # further, WITHIN the calibrated range (no recalibration needed; grip over-close is separate).
    # 1.0 = OFF.  1.3 ≈ a gentle boost (a 0.77 fist reaches full close). Only finger1 is affected.
    # Proper fix is to switch the thumb feature to a higher-range proxy + recalibrate; this is the
    # quick no-recalibration stopgap. Raise toward ~1.5 if still soft; lower if the thumb twitches.
    thumb_close_gain: float = 1.3

    # Hand (finger) command smoothing (EMA): cmd = a*target + (1-a)*prev_cmd.
    # 1.0 = OFF (raw mapper output → jittery fingers). Lower = smoother but laggier.
    # The landmark→motor mapping has no other low-pass, so this is the main anti-jitter knob
    # for "fingers feel rough / twitchy". 0.4–0.6 is a good range.
    hand_smoothing_alpha: float = 0.5

    # Per-side finger→motor-pair permutation. The mapper binds human fingers to robot motor-ID
    # pairs identically for both hands: thumb→IDs 1-2 (finger1),
    # index→IDs 3-4 (finger2), middle→IDs 5-6 (finger3), ring→IDs 7-8 (finger4). The pinky is no
    # longer tracked. This remap re-routes the computed values for the LEFT hand: the motor pair
    # physically numbered finger p receives the value computed for finger `remap[p-1]`.
    #   (1,2,3,4) = identity (no change).  (4,3,2,1) = full reverse.
    # The identity default gives the LEFT hand the same human-finger→motor-ID binding as the
    # right. The
    # left AmazingHand is a mirror build, so with identity the same human finger drives the
    # mirror-position finger on the left hand. If, on the hardware, the left fingers come out
    # swapped vs what you want, edit this tuple to the permutation that matches your build.
    # The RIGHT hand is never remapped.
    left_hand_finger_remap: tuple = (1, 2, 3, 4)

    # Optional arm command smoothing (EMA): cmd = alpha * target + (1 - alpha) * prev_cmd.
    # 1.0 = OFF (raw target, matches the previously-smooth setup). Lower = smoother but laggier.
    # Leave at 1.0 unless IK-noise judder is observed.
    arm_smoothing_alpha: float = 1.0

    # Lateral (left/right) → shoulder_pan scale. 0.0 = disabled (legacy 2D mode).
    # Units: meters → meters (same as scale_x/y). 1.0 = 1:1 lateral tracking.
    # SIGN: both hands share ONE Quest world frame and both pan motors have drive_mode=0, and the
    # two arms are mounted PARALLEL (same orientation, not mirrored). Working it through, the SAME
    # sign (+1.0) makes BOTH sides track correctly: right hand → operator-right → right gripper
    # right, AND left hand → operator-left → left gripper left. So keep both +1.0. This sign ONLY
    # matters in PLANAR mode (use_full_ik=False). The earlier "right reversed" symptom came from
    # full_ik's wrist-ORIENTATION tracking, which is mirror-handed between the Quest left/right
    # hands — fix that by using planar mode, NOT by flipping this sign.
    # Magnitude controls lateral sensitivity; 1.0 is 1:1. The signs below match the reference
    # parallel mounting and shared Quest frame. Flip both signs together if lateral motion is
    # reversed, or reduce the magnitude to use a larger human-hand envelope.
    planar_scale_z_right: float = -1.0
    planar_scale_z_left: float = -1.0

    # Gravity feed-forward for the lift joints (degrees). The arm + AmazingHand weight makes
    # shoulder_lift (ID2) / elbow_flex (ID3) sag below the commanded angle under a P-gain we
    # must keep low (high P trips the STS3215 over-torque protection — see config_so101_amazinghand).
    # This biases the COMMANDED shoulder_lift up by a few degrees so the sagged actual pose matches
    # intent without raising P.
    # +deg = command arm higher. 0 = off. Start around 3-5 and increase until the sag is gone.
    lift_feedforward_deg: float = 4.0

    # Full 3D IK: use solve_tcp_ik() to control all 5 arm joints including wrist_flex / wrist_roll.
    # use_full_ik=True:  all 5 joints controlled.
    # use_full_ik=False: planar IK (shoulder_lift + elbow_flex only, wrist joints held at start).
    #
    # wrist_orientation_scale: how much Quest wrist rotation maps to robot TCP orientation.
    #   0.0 = wrist joints auto-compensate to keep start palm orientation (no wrist tracking).
    #   1.0 = full orientation tracking.
    #  -1.0 = full tracking with inverted direction (flip if wrist_roll or wrist_flex is reversed).
    #
    # Frame alignment applied internally:
    #   Quest X (depth toward robot) → Robot X (radial)
    #   Quest Y (up)                 → Robot Z (up)
    #   Quest Z (lateral)            → Robot -Y  ← sign: Quest Z+ = Robot Y-, may need scale=-1
    use_full_ik: bool = False
    wrist_orientation_scale_right: float = 1.0
    wrist_orientation_scale_left: float = 1.0
    # fix_wrist_flex=True: wrist_flex (ID4) held at start value; only wrist_roll tracks Quest.
    # Recommended: True. Frees vertical IK from wrist compensation torque.
    fix_wrist_flex: bool = True

    # AmazingHand fingertip offset (meters) from the SO-101 wrist FLANGE, along the gripper axis.
    # The 2-link IK natively places the wrist FLANGE, but you teleoperate the FINGERTIP, which sits
    # ~0.10 m beyond it. With a nonzero offset the planar controller targets the FINGERTIP: it pulls
    # the flange back along the (pitch-held) gripper axis so the fingertip — not the flange — lands
    # where your hand is. This is what makes LATERAL (pan) tracking 1:1 too: panning rotates the long
    # hand, so without this correction the fingertip overshoots sideways (~1.37×). Engagement stays
    # smooth at ANY value (the start anchor and the pull-back use the same offset, cancelling at zero
    # delta), so a rough value already helps. MEASURE flange→fingertip on your hardware and set it;
    # 0.10 is a starting estimate. Set 0.0 to control the flange only (lateral will overshoot).
    tcp_offset_m: float = 0.10

    # Hold the gripper PITCH constant in planar / elbow-IK mode.
    # The planar IK commands the wrist FLANGE position, but you control the AmazingHand FINGERTIP,
    # which sticks out ~10 cm beyond the flange. If wrist_flex is held at a fixed ANGLE (the old
    # behaviour), then as shoulder_lift + elbow_flex move, the gripper PITCH (= wrist_flex +
    # shoulder + elbow) changes, the long hand swings, and the fingertip lands a few cm off where
    # you moved your hand. Holding pitch CONSTANT instead (wrist_flex is set
    # each tick to wrist_flex = start_pitch − shoulder_lift − elbow_flex) keeps the hand offset a
    # CONSTANT vector, so the fingertip tracks your hand more directly. The
    # gripper stays at its start orientation (level) as the arm reaches, which is also more natural
    # for manipulation. Set False to restore the old fixed-wrist_flex-angle behaviour.
    hold_gripper_pitch: bool = True

    # Elbow-retargeting IK: decouple shoulder_lift (ID2) and elbow_flex (ID3).
    # shoulder_lift ← estimated elbow position
    # elbow_flex    ← wrist-to-elbow forearm direction angle
    # forearm_axis: Quest wrist local axis pointing FROM wrist TOWARD elbow.
    #   Default [0,-1,0] = -Y. Calibrate empirically: hold arm horizontal, check estimated elbow.
    # forearm_length_m: human forearm length in meters.
    use_elbow_ik: bool = False
    forearm_axis: tuple = (0.0, -1.0, 0.0)
    forearm_length_m: float = 0.25

    stale_timeout_s: float = 3.0  # seconds before treating input as stale (Quest can freeze ~1-2s)
    baseline_samples: int = 20  # wrist samples to average for baseline

    # Startup behaviour
    move_to_start: bool = True  # auto-move robot to saved start positions on session start
    move_to_start_steps: int = 20  # interpolation steps for move-to-start
    move_to_start_delay_s: float = 0.05  # sleep between interpolation steps

    # Which side(s) this teleoperator drives. "both" (default) = full dual_arm.
    # "left" / "right" = SINGLE-SIDE mode: only that side is waited for, baselined, and emitted in
    # action_features / get_action (matching a single-arm follower with sides=…). The other Quest
    # hand is ignored entirely. In single-side mode that one side is implicitly required regardless
    # of require_both_sides.
    active_sides: str = "both"

    # Whether to require BOTH hands before proceeding.
    # Set to False to allow single-hand (left-only) operation if right is not tracked.
    # (active_sides="left"/"right" already implies single-side, so this can stay at its default.)
    require_both_sides: bool = True

    # Swap which Quest hand drives which logical side. When True, the Quest "right" hand is routed
    # to the LEFT side state (→ l_* joints → port_l + hand_port_l) and vice versa. This swaps the
    # WHOLE side together — arm joints, hand motors, calibration and startup pose — at one point
    # (handle_hts_line), so it stays coherent (unlike swapping only the arm PORTS, which splits the
    # arm from the hand). Use this to flip left/right cleanly. Default False (no swap).
    swap_sides: bool = False

    # ── Quest wrist-position jitter filter (One-Euro, Casiez et al. 2012) ──────────────────────
    # Meta Quest hand tracking is noisy: the wrist POSITION jitters by mm–cm even when the hand is
    # held still, and that jitter flows straight into the planar IK target → the arm trembles and
    # fine/precise holds are impossible. The One-Euro filter low-passes the wrist position with a
    # SPEED-ADAPTIVE cutoff: heavy smoothing when the hand is slow/still (kills jitter → rock-steady
    # holds), light smoothing when the hand moves fast (stays responsive → almost no added lag). It
    # is applied at ingest (handle_hts_line) so BOTH the live target and the captured baseline use
    # the cleaned signal. This is the single biggest knob for "trembly / imprecise" teleop.
    wrist_filter_enabled: bool = True
    # min_cutoff (Hz): the cutoff at zero speed. LOWER = smoother when still (more jitter removed)
    # but slightly more lag on slow moves. 1.0 is a good start; drop to ~0.5 if it still trembles.
    wrist_filter_min_cutoff: float = 1.0
    # beta: speed coupling. HIGHER = the cutoff opens up faster as the hand speeds up = less lag on
    # fast moves (but less jitter rejection mid-move). Lower it if holds still tremble; raise it if
    # fast moves feel laggy.
    wrist_filter_beta: float = 2.0

    # Seconds to count down before auto-capturing the wrist baseline.
    # 0.0 = wait for ENTER (manual mode, original behaviour).
    # >0  = show countdown; ENTER skips the remaining wait and captures immediately.
    baseline_countdown_s: float = 10.0

    # Debug / receiver
    print_debug: bool = False
    start_receiver: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"arm-only", "hand-only", "arm-and-hand"}:
            raise ValueError("--teleop.mode must be arm-only, hand-only, or arm-and-hand.")
        if self.stale_timeout_s <= 0:
            raise ValueError("--teleop.stale-timeout-s must be > 0.")
        if self.max_controlled_joint_step <= 0:
            raise ValueError("--teleop.max-controlled-joint-step must be > 0.")
        if self.baseline_samples <= 0:
            raise ValueError("--teleop.baseline-samples must be > 0.")
        if self.move_to_start_steps <= 0:
            raise ValueError("--teleop.move-to-start-steps must be > 0.")
