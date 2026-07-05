import uuid
import struct
from typing import Dict, List, Tuple
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage


class ImageFramesChannel(SideChannel):
    """Python 侧图像帧接收通道。
    与 Unity 侧 ImageFramesSideChannel 使用相同的 GUID 前缀 + agentId 后缀策略。

    消息整体为一个二进制串（Unity 侧通过 OutgoingMessage.SetRawBytes 写入）。
    我们不使用 IncomingMessage 的逐字段读取接口，而是一次性拿到全部 raw bytes 后自行解析，
    避免其内部读指针无法回退的问题。

    格式 (顺序、小端 32bit int)：
        int agent_id
        int camera_count
          重复 camera_count 次：
            int camera_index
            int frame_count
              重复 frame_count 次：
                int width
                int height
                int grayscale (0/1)
                int data_length
                byte[data_length] (PNG 编码)

    解析后存入 self.last_payload:
        {
            'agent_id': int,
            'cameras': {camera_index: [ (width, height, grayscale_bool, png_bytes), ... ]}
        }
    """

    BASE_PREFIX = "732c0a70-4f87-11ea-a6bf-"  # 与 Unity 侧保持一致

    def __init__(self, agent_id: int = 0):
        guid_str = self.BASE_PREFIX + format(agent_id, "012x")
        super().__init__(uuid.UUID(guid_str))
        self.agent_id = agent_id
        self.last_payload: Dict = {}

    def on_message_received(self, msg: IncomingMessage) -> None:
        raw = msg.get_raw_bytes()
        offset = 0
        def read_int() -> int:
            nonlocal offset
            if offset + 4 > len(raw):
                raise ValueError("Unexpected end of message while reading int")
            val = struct.unpack_from('<i', raw, offset)[0]
            offset += 4
            return val

        payload = {}
        try:
            agent_id = read_int()
            cam_count = read_int()
            cameras: Dict[int, List[Tuple[int, int, bool, bytes]]] = {}
            for _ in range(cam_count):
                cam_idx = read_int()
                frame_count = read_int()
                frame_list: List[Tuple[int, int, bool, bytes]] = []
                for _f in range(frame_count):
                    w = read_int()
                    h = read_int()
                    gray_flag = read_int() == 1
                    data_len = read_int()
                    if data_len < 0:
                        raise ValueError("Negative data_len")
                    if offset + data_len > len(raw):
                        raise ValueError("Unexpected end of message while reading frame bytes")
                    png_bytes = raw[offset: offset + data_len]
                    offset += data_len
                    frame_list.append((w, h, gray_flag, png_bytes))
                cameras[cam_idx] = frame_list
            payload['agent_id'] = agent_id
            payload['cameras'] = cameras
        except Exception as e:
            payload = {'error': f'ImageFramesChannel parse error: {e}'}

        self.last_payload = payload

    def get_and_clear(self):
        data = self.last_payload
        self.last_payload = {}
        return data
