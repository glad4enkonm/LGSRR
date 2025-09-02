benchmarks = {
    'MIntRec2.0': {
        'intent_labels': [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ],
        'max_seq_lengths': {
            'text': 50, 
            'video': 180, 
            'audio': 400,
        },
    },
    
    'IEMOCAP-DA':{
        'intent_labels': [
                    'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion', 
                    'Apology', 'Command', 'Agreement', 'Disagreement', 
                    'Acknowledge', 'Backchannel', 'Others'
        ],
        'label_maps': {
                    'g': 'Greeting', 'q': 'Question', 'ans': 'Answer', 'o': 'Statement Opinion', 's': 'Statement Non Opinion', 
                    'ap': 'Apology', 'c': 'Command', 'ag': 'Agreement', 'dag': 'Disagreement', 
                    'a': 'Acknowledge', 'b': 'Backchannel', 'oth': 'Others'
        },
        'max_seq_lengths': {
                'text': 44,
                'video': 230, # mean+sigma 
                'audio': 380
        },
    },
}
