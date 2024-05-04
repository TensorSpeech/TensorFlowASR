import fire


def run(
    component,
    command=None,
    name=None,
):
    """
    Run a component with cli, the helps is printed in stdout
    as https://github.com/google/python-fire/issues/188#issuecomment-791972163

    Args:
        component: functions or class
        command (optional): any. Defaults to None.
        name (str, optional):. Defaults to None.
    """
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(component, command=command, name=name)
