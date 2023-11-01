import React from "react";
import ReactPlayer from "react-player";

function VideoPlayer({ videoURL }) {
  //   videoURL =
  //     "http://localhost:5000/getVideo/static/predicted/predicted_Kids were pumped to be next to Ronaldo pregame https___t.co_1zuzjmg3GR ( 720 X 720 )_compressed.mp4";
  return (
    <div>
      <ReactPlayer url={videoURL} controls width="100%" height="370px" />
    </div>
  );
}

export default VideoPlayer;
