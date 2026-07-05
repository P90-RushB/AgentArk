from mlagents_envs.environment import UnityEnvironment
import os
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

temp_script = r'''
    using UnityEngine;
    class Example: MonoBehaviour
    {
        void SayHello()
        {
            Debug.Log("Hello ");
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = Vector3.right;
        }
    }
    '''

temp_script = r'''
using UnityEngine;
class Example: MonoBehaviour
{
    GameObject cube;
    public float rotationSpeed = 20f;
    void Start()
    {
        cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.position = Vector3.zero;

        Renderer renderer = cube.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = Color.red;
        renderer.material = material;
    }

    void Update()
    {
        cube.transform.Rotate(0, -rotationSpeed * Time.deltaTime, 0, Space.Self);
    }
}
'''

# Create the channel
string_log = StringLogChannel()

# We start the communication with the Unity Editor and pass the string_log side channel as input
env_path = os.environ.get('AGENTARK_ENV_PATH')
# env_path = None connects to the Unity Editor.

# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[string_log])
env.reset()
string_log.send_string("The environment was reset")

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
for i in range(10000):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    # We send data to Unity : A string with the number of Agent at each
    # string_log.send_string(
    #     f"Step {i} occurred with {len(decision_steps)} deciding agents and "
    #     f"{len(terminal_steps)} terminal agents"
    # )
    string_log.send_string(temp_script)

    env.step()  # Move the simulation forward

env.close()
