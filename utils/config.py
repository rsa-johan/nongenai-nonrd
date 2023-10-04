import yaml
def get_config():
    data = None
    with open('config.yml', 'r') as f:
        stream = f.read()
        data = yaml.load(stream, yaml.Loader)
    return data
