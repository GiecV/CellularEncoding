import pybullet as p
import pybullet_data
import time


def compute_fitness(individual):
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load the plane and the robot
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

    # Apply the individual's parameters to the robot
    # This part depends on how the individual's parameters are represented
    # For example, if the individual is a list of joint angles:
    for i, joint_angle in enumerate(individual):
        p.setJointMotorControl2(
            robotId, i, p.POSITION_CONTROL, targetPosition=joint_angle)

    # Simulate for a fixed amount of time
    simulation_time = 5  # seconds
    start_time = time.time()
    while time.time() - start_time < simulation_time:
        p.stepSimulation()
        time.sleep(1./240.)

    # Calculate fitness based on the distance traveled
    endPos, _ = p.getBasePositionAndOrientation(robotId)
    # Assuming fitness is based on the x-axis distance
    distance_traveled = endPos[0]

    # Disconnect from PyBullet
    p.disconnect()

    return distance_traveled


# Example usage
individual = [0.1, 0.2, 0.3, 0.4]  # Example individual parameters
fitness = compute_fitness(individual)
print(f"Fitness: {fitness}")
