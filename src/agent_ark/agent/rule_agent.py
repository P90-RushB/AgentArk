from sqlite3 import Time
import string
from .base_agent import BaseAgent

'''基于规则的agent， demo
'''
class RuleAgent(BaseAgent):

    def __init__(self, name="RuleAgent"):
        super().__init__(name)

    def forward(self, obs):
        # with open('act1.txt', 'r') as f:
        #     return f.read()

        code_act = r'''
using UnityEngine;

// Prints "waca" to the Unity console every second.
public class ArkAction_PrintWaca : MonoBehaviour
{
    int i = 1;
    private float timer = 0f;
    private const float interval = 1f;

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= interval)
        {
            var ii = 10 / i;
            Debug.Log(ii);
            Debug.Log("waca");
            timer = 0f;
        }
    }
}
// Code saved to code_act.cs

'''
        # return {k: code_act if not done.get(k, False) else '' for k in obs.keys()}
        return {k: code_act if not v['skip_infer'] else None for k, v in obs.items()}

        code_act = r'''
            using UnityEngine;
            class Example: MonoBehaviour
            {
                public RLTask rlTask;
                Vector3 controlSignal;
                float forceMultiplier = 10;
                // start
                void Start()
                {
                    Debug.Log("555");
                    rlTask = GetComponent<RLTask>();
                    controlSignal = new Vector3(1f, 0f, 1f);
                }
                public void FixedUpdate()
                {
                    SetAction();
                }
                void SetAction()
                {
                    rlTask.rBody.AddForce(controlSignal * forceMultiplier);
                }
            }
        '''

        return {k: code_act for k in obs.keys()}

        return r'''
            using UnityEngine;
            class Example: MonoBehaviour
            {
                GameObject cube;
                public float rotationSpeed = 20f;

                // start
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

'''
using UnityEngine;
public class ArkAction_PrintWaca : MonoBehaviour
{
    void Start()
    {
        var rltask = GetComponent<RLBaseTask>();
        string cmd = string.Format("U3,L6");
        rltask.ExecutePlan(cmd);
    }
}
'''
