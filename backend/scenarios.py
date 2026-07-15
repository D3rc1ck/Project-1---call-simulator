"""Call scenarios and the in-memory session engine.

Each scenario casts the simulator as the person on the other end of the line
(a customer, an interviewer, ...) and the user practices handling the call.

The engine is deliberately simple and fully offline:
- keyword rules fire once each when the user's words match, without
  consuming the script;
- otherwise the scenario advances to its next scripted line;
- when the script runs out, the closing line is spoken and the call ends.
"""

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    id: str
    keywords: tuple[str, ...]
    reply: str


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    description: str
    role: str          # who the user plays
    voice: str         # default Kokoro voice for the simulated caller
    greeting: str
    script: tuple[str, ...]
    rules: tuple[Rule, ...]
    closing: str


SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in [
        Scenario(
            id="billing_complaint",
            title="Angry customer — billing complaint",
            description="An upset customer was double-charged. You are the "
                        "support agent: calm them down and resolve the issue.",
            role="Customer support agent",
            voice="af_bella",
            greeting="Hi, yes, finally! I've been on hold forever. I just "
                     "checked my bank statement and you charged me twice this "
                     "month. Twice! I want this fixed right now.",
            script=(
                "It's forty-nine ninety-nine, charged on the third and again "
                "on the fifth. Same amount, same description. How does that "
                "even happen?",
                "Okay... and how long is that going to take? Because last "
                "time I was promised a refund it took three weeks.",
                "Fine. Can you send me an email confirmation with a case "
                "number? I want something in writing.",
                "Alright. I appreciate you actually listening, unlike the "
                "last person I talked to.",
            ),
            rules=(
                Rule(
                    id="apology",
                    keywords=("sorry", "apolog", "understand how"),
                    reply="Well... thank you for saying that. It's just "
                          "really frustrating, you know? So what are you "
                          "going to do about it?",
                ),
                Rule(
                    id="refund",
                    keywords=("refund", "reverse", "money back", "credit"),
                    reply="A refund, good. I want the full duplicate charge "
                          "back, not some account credit I'll never use.",
                ),
                Rule(
                    id="escalate",
                    keywords=("supervisor", "manager", "escalate"),
                    reply="I don't need a supervisor if you can actually fix "
                          "it yourself. Can you?",
                ),
            ),
            closing="Okay, thank you for sorting that out. I'll watch for "
                    "that email. Goodbye.",
        ),
        Scenario(
            id="job_interview",
            title="Phone screen — job interview",
            description="A recruiter is running a first-round phone screen "
                        "for a role you applied to. You are the candidate.",
            role="Job candidate",
            voice="am_michael",
            greeting="Hi, thanks for taking the time today. I'm Michael from "
                     "the recruiting team. Before we dive in — could you "
                     "briefly introduce yourself and what you're doing "
                     "currently?",
            script=(
                "Great, thanks. What made you interested in this particular "
                "role and our company?",
                "Interesting. Can you walk me through a project you're proud "
                "of, and what your specific contribution was?",
                "Good. Tell me about a time something went wrong at work — "
                "how did you handle it?",
                "Almost done. What are your salary expectations, roughly?",
                "That's helpful, thank you. Do you have any questions for me "
                "about the role or the team?",
            ),
            rules=(
                Rule(
                    id="team",
                    keywords=("team size", "how big", "who would i work"),
                    reply="Good question — the team is currently eight "
                          "people, and you'd report to the engineering lead. "
                          "Anything else you'd like to know?",
                ),
                Rule(
                    id="remote",
                    keywords=("remote", "work from home", "hybrid", "office"),
                    reply="We're hybrid — two days in the office, the rest "
                          "flexible. Does that work for you?",
                ),
            ),
            closing="Perfect, that's everything I needed. You'll hear back "
                    "from us within a week about next steps. Thanks again, "
                    "and have a great day!",
        ),
        Scenario(
            id="tech_support",
            title="Tech support — internet is down",
            description="A non-technical customer's internet stopped working. "
                        "You are the help-desk technician guiding them.",
            role="Help-desk technician",
            voice="bf_emma",
            greeting="Oh hello — I hope you can help me. My internet has "
                     "completely stopped working since this morning and I "
                     "have a video call in an hour. Nothing loads at all.",
            script=(
                "Let me look... there's a little box with lights on it under "
                "the telly. The light that's usually blue is blinking orange "
                "now. Is that bad?",
                "Alright, I've pulled the plug out. How long should I wait "
                "before putting it back in?",
                "Okay, it's plugged back in... the lights are coming on one "
                "by one... now the blue light is back and it's steady!",
                "Let me try... yes! The webpage loads. Oh, wonderful.",
            ),
            rules=(
                Rule(
                    id="restart",
                    keywords=("restart", "reboot", "unplug", "power cycle",
                              "turn it off"),
                    reply="Unplug it? The whole box? Alright, if you say so — "
                          "which cable should I pull out?",
                ),
                Rule(
                    id="lights",
                    keywords=("light", "led", "blinking", "colour", "color"),
                    reply="The lights, right. There are four of them. The "
                          "second one is the one blinking orange.",
                ),
            ),
            closing="Thank you so much, you've saved my morning. I'll write "
                    "down that unplugging trick for next time. Goodbye!",
        ),
    ]
}

FALLBACK_LINE = "Sorry, I didn't quite catch that — could you say it again?"


@dataclass
class CallSession:
    scenario: Scenario
    voice: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    step: int = 0
    used_rules: set[str] = field(default_factory=set)
    ended: bool = False

    def next_reply(self, user_text: str) -> tuple[str, bool]:
        """Return (reply_text, call_ended) for the user's utterance."""
        if self.ended:
            return self.scenario.closing, True

        lowered = user_text.lower()
        for rule in self.scenario.rules:
            if rule.id in self.used_rules:
                continue
            if any(kw in lowered for kw in rule.keywords):
                self.used_rules.add(rule.id)
                return rule.reply, False

        if self.step < len(self.scenario.script):
            reply = self.scenario.script[self.step]
            self.step += 1
            return reply, False

        self.ended = True
        return self.scenario.closing, True


class SessionStore:
    """In-memory store; sessions live for the duration of the process."""

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    def create(self, scenario_id: str, voice: str | None = None) -> CallSession:
        scenario = SCENARIOS.get(scenario_id)
        if scenario is None:
            raise KeyError(scenario_id)
        session = CallSession(scenario=scenario, voice=voice or scenario.voice)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> CallSession | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
