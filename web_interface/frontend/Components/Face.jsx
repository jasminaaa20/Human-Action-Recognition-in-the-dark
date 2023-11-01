import React, { useState } from "react";

function extractTimeFromName(imageName) {
  const nameParts = imageName.split("_");
  const timeString = nameParts[nameParts.length - 1].split(".")[0]; // Remove the file extension

  const hours = timeString.slice(0, 2);
  const minutes = timeString.slice(2, 4);
  const seconds = timeString.slice(4, 6);
  return `${hours}:${minutes}:${seconds}`;
}

export default function Face({ imageUrl, serverURL, baseFolder, name }) {
  const [hoveredTime, setHoveredTime] = useState(null);

  const handleImageHover = () => {
    setHoveredTime(extractTimeFromName(imageUrl));
  };

  const handleImageLeave = () => {
    setHoveredTime(null);
  };

  return (
    <div
      className=" p-2  flex flex-col items-center justify-center relative"
      onMouseEnter={handleImageHover}
      onMouseLeave={handleImageLeave}
    >
      <img
        src={`${serverURL}/getImage/${baseFolder}/${name}/${imageUrl}`}
        alt="Image"
        className="w-12 h-12 rounded-md"
      />
      {hoveredTime && (
        <p className="text-sm absolute -bottom-6 left-1/2 transform -translate-x-1/2 whitespace-nowrap bg-gray-500 text-white p-1 rounded-xl">
          Seen At {hoveredTime}
        </p>
      )}
    </div>
  );
}
