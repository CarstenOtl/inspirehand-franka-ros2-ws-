# Corporate root CAs

Drop `*.crt` files here before `docker compose build` if your network does TLS
inspection (FortiClient and similar) and the image build cannot otherwise reach
`packages.ros.org` or `github.com`.

They are machine-specific, so `.gitignore` keeps `*.crt` out of the repo. This
file exists so the directory is committed and `COPY certs/` has something to
copy — an empty glob fails a Docker build, an empty directory does not.
