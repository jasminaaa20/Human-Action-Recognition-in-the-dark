"use client";

import { useState } from "react";
import WebCam from "@/Components/WebCam";
import Faces from "@/Components/Faces";
import VideoPlayer from "@/Components/VideoPlayer";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

const serverURL = "http://localhost:5000";

export default function Home() {
  const [streaming, setStreaming] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [processing, setProcessing] = useState(false);

  const handleVideoUpload = async (event) => {
    processing;
    const file = event.target.files[0];

    setSelectedVideo(file);

    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      // Send the video to the backend for processing
      console.log("Sending video to backend for processing...");
      const response = await fetch(serverURL + "/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        console.error("Error processing video.");
      }
      const data = await response.json();
      const videoURL = data.video_url;
      setVideoURL(videoURL);
      console.log("Video URL: ", videoURL);
      setSelectedVideo(null);
    } else {
      console.error("No video selected.");
    }
    setProcessing(false);
  };

  const handleWebCam = async (isOpen) => {
    setVideoURL(null);
    setSelectedVideo(null);

    const response = await fetch(serverURL + "/toggleStream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(isOpen),
    });

    if (!response.ok) {
      console.error("Error processing video.");
    }

    setStreaming(isOpen);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-blue-500 text-white py-4 text-3xl text-center font-bold">
        Action Detection in Challanging Lighting Conditions
        <p className="text-sm text-gray-200"> CS3501: Data Science Project </p>
      </header>

      <div className="flex-col max-w-5xl mx-auto p-4 flex">
        <h2 className="text-xl text-slate-500 font-semibold mb-4">
          Choose Action
        </h2>
        <div className="flex-row flex items-stretch bg-white p-4 space-x-3 rounded-lg shadow-md">
          <div className="flex-1">
            {streaming ? (
              <button
                onClick={() => handleWebCam(false)}
                className="bg-blue-500 text-white p-2 rounded-md mb-2 block text-left w-full transition duration-300 ease-in-out transform hover:scale-105"
              >
                Action 01: Close Camera 01
              </button>
            ) : (
              <button
                onClick={() => handleWebCam(true)}
                className="bg-blue-500 text-white p-2 rounded-md mb-2 block text-left w-full"
              >
                Action 01: Open Camera 01
              </button>
            )}
          </div>
          <div className="flex-1 transition duration-300 ease-in-out transform hover:scale-105">
            <input
              type="file"
              accept="video/*"
              onChange={() => {
                handleWebCam(false);
                handleVideoUpload(event);
                setProcessing(true);
              }}
              id="videoInput"
              className="hidden"
            />
            <label
              htmlFor="videoInput"
              className="bg-blue-500 text-white p-2 rounded-md mb-2 block text-left cursor-pointer w-full"
            >
              {processing ? (
                <div className="animate-fade-in">
                  <FontAwesomeIcon icon={faSpinner} spin /> Video Processing...
                </div>
              ) : (
                <p>Action 02: Upload Video</p>
              )}
            </label>
          </div>
        </div>

        <div className="p-4 flex flex-row justify-center">
          {streaming ? (
            <WebCam />
          ) : videoURL ? (
            <VideoPlayer videoURL={`${serverURL}\\getVideo\\${videoURL}`} />
          ) : (
            <img
              src="https://images.squarespace-cdn.com/content/v1/609c3b2b28705e0830c8d28e/1632195322170-KVV8YK3NOBNV5U9NU3JS/New+Website+Images15.png"
              alt="Placeholder"
              width={800}
              height={600}
            />
          )}
        </div>
        <div className="flex flex-row justify-left mt-4">
          {streaming || videoURL ? (
            <Faces />
          ) : (
            <div className=" text-center text-xl text-slate-500 font-semibold ">
              Upload a video or open the camera to see the results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
