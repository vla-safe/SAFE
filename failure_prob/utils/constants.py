from collections import defaultdict
import itertools

EXP_NAME_2_DISPLAY_NAME = {
    "openvla_libero_v2": "OpenVLA LIBERO",
    "opi0_simpler_v1": "$\\pi_0^*$ SimplerEnv",
    "pi0fast_libero_v4": "$\\pi_0$-FAST LIBERO",
    "pi0diff_libero_v1": "$\\pi_0$ LIBERO",
    "pi0fast_droid_0510": "$\\pi_0$-FAST Franka",
}


METHOD_NAME_2_GROUP_ID = {
    "lstm": 0,
    "indep": 0,
    # "transformer": 0,
    
    "max_token_prob": 2,
    "avg_token_prob": 2,
    "max_token_entropy": 2,
    "avg_token_entropy": 2,
    
    "embed-mahala": 3,
    "embed-euclid": 3,
    "embed-cosine": 3,
    "embed-pca_kmeans": 3,
    
    "rnd": 4,
    "logpZO": 4,
    
    "stac_mmd": 6,
    "stac_single": 5,
    
    "total_var": 6,
    "pos_var": 6,
    "rot_var": 6,
    "gripper_var": 6,
    "entropy_linkage": 6,
}

METHOD_NAME_2_DISPLAY_NAME = {
    "max_token_prob": "Max prob",
    "avg_token_prob": "Avg prob",
    "max_token_entropy": "Max entropy",
    "avg_token_entropy": "Avg entropy",
    
    "embed-mahala": "Mahalanobis dist.",
    "embed-euclid": "Euclidean dist. k-NN",
    "embed-cosine": "Cosine dist. k-NN",
    "embed-pca_kmeans": "PCA-KMeans",
    "rnd": "RND",
    "logpZO": "LogpZO",
    
    "total_var": "Action total var.",
    "pos_var": "Trans. total var.",
    "rot_var": "Rot. total var.",
    "gripper_var": "Gripper total var.",
    "entropy_linkage": "Cluster entropy",
    
    "stac_mmd": "STAC",
    "stac_single": "STAC-Single",
    
    "lstm": "SAFE-LSTM",
    "indep": "SAFE-MLP",
}

# define as many distinct markers as you like
MARKER_TYPE_CYCLE = itertools.cycle(['o', 's', '^', 'D', 'v', 'X', '*'])
METHOD_NAME_2_MARKER = {
    k: v for k, v in zip(METHOD_NAME_2_GROUP_ID.keys(), MARKER_TYPE_CYCLE)
}


# Hand-crafted metrics for different models
MANUAL_METRICS = defaultdict(lambda: None, {
    "openvla": None,
    "openvla-multi": [
        "action/cum_total_var",
        "action/cum_general_var",
        "action/cum_pos_var",
        "action/cum_rot_var",
        "action/cum_gripper_var",
        "action/cum_entropy_linkage.01",
        "action/cum_entropy_linkage.05",
    ],
    "octo": None,
    "open_pizero": None,
    "pizero": None,
    "pizero_fast": None,
})


EVAL_TIME_QUANTILES = defaultdict(lambda: [0.25, 0.5, 0.75, 1.0], {
    "openvla": [0.25, 0.5, 0.75, 1.0],
    "openvla-multi": [0.25, 0.5, 0.75, 1.0],
    "octo": [0.25, 0.5, 0.75, 1.0],
    "open_pizero": [0.25, 0.5, 0.75, 1.0],
    "pizero": [0.25, 0.5, 0.75, 1.0],
    "pizero_fast": [0.25, 0.5, 0.75, 1.0],
})

TASK_SPLITS = {
    "openvla": {
        "default": {
            "seen": [2, 8, 4, 9, 1, 6, 7], 
            "unseen": [3, 0, 5],
        },
        "spatial": {
            "seen": [2, 8, 4, 9, 1, 6, 7], 
            "unseen": [3, 0, 5],
        },
        "goal": {
            "seen": [2, 8, 4, 9, 1, 6, 7], 
            "unseen": [3, 0, 5],
        },
        "object": {
            "seen": [2, 8, 4, 9, 1, 6, 7], 
            "unseen": [3, 0, 5],
        },
        "10": {
            "seen": [2, 8, 4, 9, 1, 6, 7], 
            "unseen": [3, 0, 5],
        },
    }
}