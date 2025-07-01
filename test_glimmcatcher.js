const axios = require('axios');

async function testGlimmcatcher() {
  try {
    const response = await axios.post('http://127.0.0.1:5001/glimmcatcher', {
      user_input: "hi there buddy ,design a mt 15 bike for me, it must have to be look realistic, with a good design,",
      user_id: "test_node_user"
    });
    console.log("Response from FastAPI:", response.data);
  } catch (err) {
    console.error("Error:", err.response ? err.response.data : err.message);
  }
}

testGlimmcatcher();