def get_threshold(env, constraint='velocity'):
    if constraint == 'safetygym':
        thresholds = {  'Safexp-PointButton1-v0': 10,
                        'Safexp-PointButton2-v0': 10,
                        'Safexp-PointGoal1-v0': 10,
                        'Safexp-PointGoal2-v0': 10,
                        'Safexp-PointPush1-v0': 10,
                        'Safexp-PointPush2-v0': 10,
                        'Safexp-CarButton1-v0': 10,
                        'Safexp-CarButton2-v0': 10,
                        'Safexp-CarGoal1-v0': 10,
                        'Safexp-CarGoal2-v0': 10,
                        'Safexp-CarPush1-v0': 10,
                        'Safexp-CarPush2-v0': 10,
                        'Safexp-DoggoButton1-v0': 10,
                        'Safexp-DoggoButton2-v0': 10,
                        'Safexp-DoggoGoal1-v0': 10,
                        'Safexp-DoggoGoal2-v0': 10,
                        'Safexp-DoggoPush1-v0': 10,
                        'Safexp-DoggoPush2-v0': 10,
                        'SafetyCarCircle-v0': 20,
                        'SafetyBallReach-v0': 10,
                        'SafetyBallCircle-v0': 20,
                        'SafetyDroneRun-v0': 20,
                        'SafetyBallReach-v0': 0.16,
                        'SafetyAntRun-v0': 10                        
                      }
    elif constraint == 'velocity':
        thresholds = {'Ant-v3': 103.115,
                      'HalfCheetah-v3': 151.989,
                      'Hopper-v3': 82.748,
                      'Humanoid-v3': 20.140,
                      'Swimmer-v3': 24.516,
                      'Walker2d-v3': 81.886
                      }
    return thresholds[env]