def get_threshold(env, constraint='velocity'):
    if constraint == 'safetygym':
        thresholds = {'Safexp-CarButton1-v0': 10,
                        'Safexp-CarButton2-v0': 10,
                        'Safexp-PointButton1-v0': 10,
                        'Safexp-PointButton2-v0': 10,
                        'Safexp-PointPush1-v0':10
                      }
    elif constraint == 'velocity':
        thresholds = {'Ant-v3': 103.115,
                      'HalfCheetah-v3': 151.989,
                      'Hopper-v3': 82.748,
                      'Humanoid-v3': 20.140,
                      }
    return thresholds[env]