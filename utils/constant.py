FEATURES = {
    'helpfulness': {
        'attribute_desc': "is helpful for the original poster",
        'attr_min': "not helpful",
        'attr_max': "very helpful",
    },
    'specificity': {
        'attribute_desc': "is specific enough",
        'attr_min': "too vague",
        'attr_max': "very specific",
    },
    'intent': {
        'attribute_desc': "understands the original poster's intent",
        'attr_min': "failure of understanding",
        'attr_max': "perfect understanding",
    },
    'factuality': {
        'attribute_desc': "is factually correct",
        'attr_min': "egregiously incorrect",
        'attr_max': "fully correct",
    },
    'easy-to-understand': {
        'attribute_desc': "is easy to understand",
        'attr_min': "very difficult to understand",
        'attr_max': "very easy to understand",
    },
    'relevance': {
        'attribute_desc': "is relevant to the original poster's question",
        'attr_min': "off-topic",
        'attr_max': "very relevant",
    },
    'readability': {
        'attribute_desc': "is easy to read and not too technical for the original poster",
        'attr_min': "very difficult to read",
        'attr_max': "very easy to read",
    },
    'enough-detail': {
        'attribute_desc': "provides enough detail to be helpful",
        'attr_min': "too little detail",
        'attr_max': "very detailed",
    },
    'biased:': {
        'attribute_desc': "is biased or one-sided",
        'attr_min': "very biased",
        'attr_max': "not biased at all",
    },
    'fail-to-consider-individual-preferences': {
        'attribute_desc': "fails to consider the original poster's cultural or individual preferences",
        'attr_min': "takes into account the original poster's preferences",
        'attr_max': "fails to consider the original poster's preferences",
    },
    'repetetive': {
        'attribute_desc': "is repetitive",
        'attr_min': "very repetitive",
        'attr_max': "not repetitive",
    },
    'fail-to-consider-context': {
        'attribute_desc': "fails to consider the original poster's context",
        'attr_min': "fails to consider the original poster's context",
        'attr_max': "takes into account the original poster's context",
    },
    'too-long': {
        'attribute_desc': "is too long",
        'attr_min': "too long",
        'attr_max': "not too long",
    },
}