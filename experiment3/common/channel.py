class Channel:
    """模拟双方消息交换，记录发送的总字节数。"""
    def __init__(self):
        self.total_bytes = 0
        self.log = []

    def send(self, sender: str, data: bytes) -> bytes:
        self.total_bytes += len(data)
        self.log.append((sender, len(data)))
        return data