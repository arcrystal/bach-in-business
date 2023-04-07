import { useState } from "react";
import "./App.css";
import axios from "axios";
import { CircularProgress } from "@mui/material";
import "./fonts/Merriweather-Regular.ttf";

function App() {
  const [midiFilename, setMidiFilename] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  function getData() {
    setIsLoading(true);
    axios({
      method: "GET",
      url: "http://127.0.0.1:5000/get_music",
    })
      .then((response: { data: { filename: any } }) => {
        setMidiFilename(response.data.filename);
        setIsLoading(false);
      })
      .catch((error: { response: { status: any; headers: any } }) => {
        if (error.response) {
          console.log(error.response);
          console.log(error.response.status);
          console.log(error.response.headers);
        }
      });
  }

  return (
    <div
      style={{
        height: "100vh",
        backgroundColor: "#82beff",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          height: "60vh",
        }}
      >
        <div className="Button" onClick={() => getData()}>
          <div>Generate</div>
        </div>

        <div hidden={!isLoading}>
          <CircularProgress />
        </div>

        {midiFilename && (
          <div>
            <p>name: {midiFilename}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
