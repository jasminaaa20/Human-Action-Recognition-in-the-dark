import React from "react";

const WebCam = () => {
  return (
    <img
      src={"http://localhost:5000" + "/stream"}
      alt="Live Stream"
      className="bg-orange-700"
      width={800}
      height={600}
    />
  );
};

export default WebCam;
