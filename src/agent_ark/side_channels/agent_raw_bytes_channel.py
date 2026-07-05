import uuid
from typing import List, Tuple

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


class AgentRawBytesChannel(SideChannel):
    """Per-agent raw-bytes channel with simple UTF-8 text protocol.

    Protocol:
      - Unity -> Python:
          "[code_act]<0|1>:<code>"  => (run_flag, code)
          "[log]<text>"            => treated as log text
          other text                => treated as log text

      - Python -> Unity:
          "[log]<text>"            => log
          "[code_act]<0|1>:<code>"  => code action

    GUID strategy matches Unity side: base prefix + 12-hex agent id.
    """

    def __init__(self, agent_id: int = 0) -> None:
        channel_uuid = AgentRawBytesChannel.get_agent_guid(agent_id)
        super().__init__(channel_uuid)
        self.agent_id = agent_id
        self.step_msgs: List[str] = []
        self.exec_flag: bool = False
        self.code_string: str = ""

    def on_message_received(self, msg: IncomingMessage) -> None:
        raw = msg.get_raw_bytes()
        flag, text = self._parse_payload(raw)
        self.exec_flag = flag
        self.code_string = text
        self.step_msgs.append(text)

    def send_string(self, data: str) -> None:
        self._send_payload("[log]", data or "")

    def send_code_act(self, run_flag: bool, code_str: str) -> None:
        body = f"[code_act]{1 if run_flag else 0}:{code_str or ''}"
        self._send_payload("", body)

    def clear_step_msgs(self) -> None:
        self.step_msgs = []

    def get_step_msgs(self) -> List[str]:
        return self.step_msgs

    def _send_payload(self, prefix: str, text: str) -> None:
        payload = (prefix + text).encode("utf-8")
        msg = OutgoingMessage()
        msg.set_raw_bytes(bytearray(payload))
        super().queue_message_to_send(msg)

    @staticmethod
    def _parse_payload(raw: bytes) -> Tuple[bool, str]:
        if not raw:
            return False, ""
        txt = raw.decode("utf-8", errors="replace")
        if txt.startswith("[code_act]"):
            body = txt[len("[code_act]") :]
            parts = body.split(":", 1)
            run_flag = parts[0] == "1" if parts else False
            code = parts[1] if len(parts) > 1 else ""
            return run_flag, code
        if txt.startswith("[log]"):
            return False, txt[len("[log]") :]
        return False, txt

    @staticmethod
    def get_agent_guid(agent_id: int) -> uuid.UUID:
        base = "621f0a70-4f87-11ea-a6bf-"
        agent_id_hex = format(agent_id, "012x")
        guid_str = base + agent_id_hex
        return uuid.UUID(guid_str)
