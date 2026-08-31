module.exports = function pnginfo(buffer) {
  // PNG IHDR chunk: bytes 16-23 are width/height (big-endian uint32)
  const width = buffer.readUInt32BE(16);
  const height = buffer.readUInt32BE(20);
  return { width, height };
};
