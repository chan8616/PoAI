def fix_validator(fixed_value=None):
    return ({} if fixed_value is None else {
        'validator': {
            'test': "user_input=='{}'".format(fixed_value),
            'message': 'Unmodifiable'}
        })
