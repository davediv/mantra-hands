"""
The cast state machine.

::

    Idle ──charge conditions──▶ Charging ──held 400 ms──▶ Armed ──push──▶ Cast
      ▲                             │                        │             │
      └───────────conditions lost───┴──────grace expired──────┘             │
      └─────────────────────────── Cooldown ◀──────────────────────────────┘

Every threshold in here is a pair: one value to enter a state and a different
value to leave it. A single threshold oscillates whenever the signal sits on the
boundary, and a rune that strobes on and off reads as a broken app rather than
as a hand held slightly wrong.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from .config import GESTURE
from .filters import TimedRing
from .landmarks import HandFrame


class Phase(IntEnum):
    IDLE = 0
    CHARGING = 1
    ARMED = 2
    COOLDOWN = 3


PHASE_NAME = ("idle", "charging", "armed", "cooldown")


def power_for(punch_rate: float) -> float:
    """Maps a short-window expansion rate onto the 0.35..1 power the burst reads."""
    t = (punch_rate - GESTURE.expand_rate_on) / (GESTURE.punch_full_rate - GESTURE.expand_rate_on)
    return 0.35 + 0.65 * max(0.0, min(1.0, t))


class CastRecognizer:
    def __init__(self) -> None:
        self.phase = Phase.IDLE

        #: How long the charge has been held, in ms.
        self.charge_ms = 0.0
        #: Charge as a 0..1 ramp for the visuals. Reaches 1 well after the arm point.
        self.charge01 = 0.0

        #: One-update pulses, valid only for the update that set them.
        self.fired = False
        self.started_charging = False
        #: 0.35..1, from how hard the push was.
        self.power = 0.0

        # Readouts for the HUD and for the double-cast test.
        self.extended_count = 0
        self.facing = 0.0
        self.palm_speed = 0.0
        self.expand_rate = 0.0
        self.punch_rate = 0.0
        self.area_growth = 1.0

        self._finger_up = np.zeros(4, dtype=np.uint8)
        self._facing_latched = False
        self._steady_latched = False
        self._expand_latched = False

        self._areas = TimedRing(16)
        self._prev_palm_x = 0.0
        self._prev_palm_y = 0.0
        self._last_ms = -1.0
        self._was_present = False
        self._grace_ms = 0.0
        self._cooldown_ms = 0.0
        self._push_rising_edge = False
        self._power_window_ms = 0.0

    @property
    def armed(self) -> bool:
        """True once the charge has been held long enough that a push would fire."""
        return self.phase == Phase.ARMED

    def update(self, hand: HandFrame, t_ms: float) -> None:
        """
        Runs once per detection, not once per rendered frame. Gestures read the
        measured landmarks; only the renderer reads the extrapolated ones. Judging
        a push from extrapolated positions would mean judging it partly from our
        own guess about it.
        """
        self.fired = False
        self.started_charging = False

        dt_ms = 0.0 if self._last_ms < 0 else min(200.0, t_ms - self._last_ms)
        self._last_ms = t_ms

        if not hand.present:
            # A hand that has left the frame carries no history worth keeping. In
            # particular the area ring must go, so that the same hand coming back
            # cannot be read as a push.
            if self._was_present:
                self.hard_reset()
            self._was_present = False
            if self.phase == Phase.COOLDOWN:
                self._cooldown_ms -= dt_ms
                if self._cooldown_ms <= 0:
                    self.phase = Phase.IDLE
            return

        # Re-acquisition. This only trips on the transition into presence.
        if not self._was_present:
            self._areas.clear()
            self._prev_palm_x = hand.palm_x
            self._prev_palm_y = hand.palm_y
            self._expand_latched = False
        self._was_present = True

        self._measure(hand, t_ms, dt_ms)
        self._step(dt_ms)
        self.charge01 = min(1.0, self.charge_ms / GESTURE.charge_full_ms)

    def _measure(self, hand: HandFrame, t_ms: float, dt_ms: float) -> None:
        """Updates every latched signal from this detection."""
        # --- fingers: four independent hysteresis latches ---
        up = 0
        for f in range(4):
            ratio = float(hand.finger_ext[f])
            was = self._finger_up[f] == 1
            now = ratio > GESTURE.extend_off if was else ratio > GESTURE.extend_on
            self._finger_up[f] = 1 if now else 0
            if now:
                up += 1
        self.extended_count = up

        # --- palm facing ---
        self.facing = hand.facing
        self._facing_latched = (
            hand.facing > GESTURE.facing_off
            if self._facing_latched
            else hand.facing > GESTURE.facing_on
        )

        # --- steadiness, in hand-widths per second ---
        if dt_ms > 0:
            dx = hand.palm_x - self._prev_palm_x
            dy = hand.palm_y - self._prev_palm_y
            moved = math.sqrt(dx * dx + dy * dy) / hand.hand_size
            speed = moved / (dt_ms / 1000.0)
            # Smoothed, or one noisy detection would break the hold.
            self.palm_speed += (speed - self.palm_speed) * 0.45
        self._prev_palm_x = hand.palm_x
        self._prev_palm_y = hand.palm_y
        self._steady_latched = (
            self.palm_speed < GESTURE.steady_off
            if self._steady_latched
            else self.palm_speed < GESTURE.steady_on
        )

        # --- push: relative growth of the hand's bounding-box area ---
        self._areas.push(t_ms, hand.area)
        self.expand_rate = self._rate_over(t_ms, hand.area, GESTURE.area_window_ms)
        self.punch_rate = self._rate_over(t_ms, hand.area, GESTURE.punch_window_ms)
        self.area_growth = 1.0 + self.expand_rate * (GESTURE.area_window_ms / 1000.0)

        pushing = self.area_growth > GESTURE.min_area_growth and (
            self.expand_rate > GESTURE.expand_rate_off
            if self._expand_latched
            else self.expand_rate > GESTURE.expand_rate_on
        )
        was_pushing = self._expand_latched
        self._expand_latched = (
            self.expand_rate > GESTURE.expand_rate_off if self._expand_latched else pushing
        )
        self._push_rising_edge = self._expand_latched and not was_pushing

        # Power keeps being revised upward for a moment after the cast fires, as
        # the rest of the push lands. See GESTURE.power_settle_ms.
        if self._power_window_ms > 0:
            self._power_window_ms -= dt_ms
            p = power_for(self.punch_rate)
            if p > self.power:
                self.power = p

    def _rate_over(self, t_ms: float, area: float, window_ms: float) -> float:
        """
        Relative growth rate of the bounding-box area over ``window_ms``, per second.

        Returns 0 until the ring actually reaches back that far. That guard is
        what stops a hand entering the frame from registering as a cast: for the
        first window's worth of a newly seen hand there is simply nothing to
        compare against, so there is no rate to trip on.
        """
        target = t_ms - window_ms
        if self._areas.oldest_time < 0 or self._areas.oldest_time > target:
            return 0.0
        then = self._areas.sample_at(target)
        if then <= 1e-6:
            return 0.0
        return (area / then - 1.0) / (window_ms / 1000.0)

    def _step(self, dt_ms: float) -> None:
        holding = (
            self.extended_count >= GESTURE.extend_count
            and self._facing_latched
            and self._steady_latched
        )

        if self.phase == Phase.IDLE:
            if holding:
                self.phase = Phase.CHARGING
                self.charge_ms = 0.0
                self.started_charging = True

        elif self.phase == Phase.CHARGING:
            if not holding:
                self.phase = Phase.IDLE
                self.charge_ms = 0.0
                return
            self.charge_ms += dt_ms
            if self.charge_ms >= GESTURE.charge_min_ms:
                self.phase = Phase.ARMED
                self._grace_ms = GESTURE.arm_grace_ms

        elif self.phase == Phase.ARMED:
            self.charge_ms += dt_ms
            if holding:
                self._grace_ms = GESTURE.arm_grace_ms
            else:
                # A hard push blurs the hand and the landmarker's confidence dips
                # for a frame or two, which briefly breaks the hold test. Without
                # this window the user's most committed push is the one that fails.
                self._grace_ms -= dt_ms
                if self._grace_ms <= 0:
                    self.phase = Phase.IDLE
                    self.charge_ms = 0.0
                    return
            if self._push_rising_edge:
                self.fired = True
                self.power = power_for(self.punch_rate)
                self._power_window_ms = GESTURE.power_settle_ms
                self.phase = Phase.COOLDOWN
                self._cooldown_ms = GESTURE.cooldown_ms
                self.charge_ms = 0.0

        elif self.phase == Phase.COOLDOWN:
            self._cooldown_ms -= dt_ms
            # The push has to have finished as well as the clock having run out,
            # or a long shove rolls straight into the next charge.
            if self._cooldown_ms <= 0 and not self._expand_latched:
                self.phase = Phase.IDLE

    def hard_reset(self) -> None:
        """Drops every latch and all history. Used when the hand leaves the frame."""
        self.phase = Phase.IDLE
        self.charge_ms = 0.0
        self.charge01 = 0.0
        self.fired = False
        self.expand_rate = 0.0
        self.punch_rate = 0.0
        self.area_growth = 1.0
        self.palm_speed = 0.0
        self._power_window_ms = 0.0
        self._finger_up.fill(0)
        self._facing_latched = False
        self._steady_latched = False
        self._expand_latched = False
        self._push_rising_edge = False
        self._areas.clear()
        self._grace_ms = 0.0
        self._cooldown_ms = 0.0

    @property
    def phase_name(self) -> str:
        return PHASE_NAME[self.phase]
