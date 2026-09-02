# Uploading this repository to GitHub

The package is prepared for the GitHub account **Arthur-KI** and assumes the
repository name `symbolic-u`.

## Easiest: GitHub web interface

1. Create a new empty repository named `symbolic-u` under `Arthur-KI`.
2. Do **not** ask GitHub to generate a README, `.gitignore` or license, because
   those files already exist here.
3. Extract the provided ZIP locally.
4. Open the extracted `Arthur-KI-symbolic-u` directory.
5. Upload **the contents of that directory**, not the outer ZIP file.
6. Commit the upload to `main`.
7. Open the Actions tab. The included `CI` workflow should run automatically.
8. Add the description/topics from `REPOSITORY_METADATA.md` if desired.

GitHub does not automatically unpack an uploaded ZIP into a repository tree, so
extract it before using the web uploader.

## Git command line

From inside the extracted directory:

```bash
git init
git branch -M main
git add .
git commit -m "Initial public Symbolic-U research release"
git remote add origin https://github.com/Arthur-KI/symbolic-u.git
git push -u origin main
```

If Git asks for authentication, use GitHub's normal browser/credential-manager or
SSH flow rather than putting a password/token into the repository.

## After upload

Recommended checks:

- CI is green;
- GitHub displays the Apache-2.0 license;
- GitHub shows the `Cite this repository` control from `CITATION.cff`;
- `research/archive_full/INDEX.md` renders correctly;
- the repository remains well below GitHub's normal size limits.
