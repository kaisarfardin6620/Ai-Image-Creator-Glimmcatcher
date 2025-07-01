const axios = require('axios');

async function testGlimmcatcher() {
  try {
    const response = await axios.post('http://127.0.0.1:5001/glimmcatcher', {
      user_input: "can u help me make an image 'a girl with blonde hair and blue eyes sitting on a bench in a park, holding wine glass looking at the sky, 4k reslotuion 1024*1024'",
      user_id: "test_node_user"
    });
    console.log("Response from FastAPI:", response.data);
  } catch (err) {
    console.error("Error:", err.response ? err.response.data : err.message);
  }
}

testGlimmcatcher();