class IllegalConfigurationError(Exception):
    """Configuration is not valid.
    The most typical case is the configuration file is generated,
    but the file is not property configured.
    """
