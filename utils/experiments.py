import argparse
import datetime
import json
import os
import sys
import tempfile


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S.%f'):
    """
    Largely a copy of chainerRLs prepare output dir
    See (https://github.com/chainer/chainerrl/blob/018a29132d77e5af0f92161250c72aba10c6ce29/chainerrl/experiments/prepare_output_dir.py)
    Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables

    Args:
        args (dict or argparse.Namespace): Arguments to save
        user_specified_dir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """
    time_str = datetime.datetime.now().strftime(time_format)
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        if argv is None:
            argv = sys.argv
        f.write(' '.join(argv))

    print('Results stored in {:s}'.format(os.path.abspath(outdir)))
    return outdir
