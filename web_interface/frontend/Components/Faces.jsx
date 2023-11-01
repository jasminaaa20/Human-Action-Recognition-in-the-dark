import React, { useState, useEffect } from "react";
import Face from "@/Components/Face";

const serverURL = "http://localhost:5000";

function Faces() {
  const [imageUrls, setImageUrls] = useState([]);
  const [baseFolder, setBaseFolder] = useState("");

  const fetchData = () => {
    fetch(serverURL + "/getFaces")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        const images = data?.images;
        setBaseFolder(data?.base_folder);
        setImageUrls(images || []);
      })
      .catch((error) => {
        console.error("Error fetching image URLs:", error);
      });
  };

  useEffect(() => {
    // Fetch data every 5 seconds
    const intervalId = setInterval(() => {
      fetchData();
    }, 10000);

    // Cleanup the interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="bg-gray-100 min-h-screen p-4">
      {imageUrls.length === 0 ? (
        <p className="text-xl text-slate-500 font-semibold ">
          Detected faces will show here.
        </p>
      ) : (
        <>
          <h1 className="flex text-3xl font-bold text-center mb-8">
            Face Gallery
          </h1>
          <div className="grid grid-cols-2">
            {Object.keys(imageUrls).map((name) => (
              <div
                key={name}
                className="bg-white rounded-lg shadow-lg p-2 mb-4 mr-3"
              >
                <h2 className="text-xl font-semibold">{name}</h2>
                <div className="grid grid-cols-5 gap-4">
                  {imageUrls[name]?.map((imageUrl, index) => (
                    <Face
                      key={index}
                      imageUrl={imageUrl}
                      serverURL={serverURL}
                      baseFolder={baseFolder}
                      name={name}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default Faces;
