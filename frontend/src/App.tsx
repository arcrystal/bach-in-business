import { useState } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [musicData, setMusicData] = useState(null);

  function getData() {
    axios({
      method: "GET",
      url: "http://127.0.0.1:5000/get_music",
    })
      .then((response: { data: { name: any } }) => {
        const res = response.data.name;
        alert(res);
        setMusicData(res);
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
    <div className="App">
      <header className="App-header">
        <div onClick={() => getData()}>Press</div>
        {musicData && (
          <div>
            <p>name: {musicData}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
