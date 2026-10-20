# GitHub Repo & Projects

## A quick guide for Monday's deadline

**Goal:** Work together safely, track tasks, and submit reviewed changes.

---

## 1. The key ideas

- **Repository (repo):** The shared project files and their history.
- **Branch:** A separate workspace for one task or feature.
- **Issue:** A task, bug, or question.
- **Pull request (PR):** A request to review and merge code.
- **Project:** A board that organizes issues and progress.

**Simple flow:** Issue → Branch → Commit → Push → Pull request → Review → Merge

---

## 2. Start with the repository

1. Open the team repository on GitHub.
2. Clone it to your computer:

```bash
git clone https://github.com/TEAM/REPOSITORY.git
cd REPOSITORY
```

3. Update your local copy before starting:

```bash
git switch main
git pull origin main
```

---

## 3. Work on a branch

Never make team changes directly on `main`.

```bash
git switch -c feature/short-task-name
# edit files
git add .
git commit -m "Describe the change"
git push -u origin feature/short-task-name
```

**Tip:** Use one branch per task or issue.

---

## 4. Open and review a Pull Request

On GitHub:

1. Open a PR from your branch into `main`.
2. Link the related Issue, for example: `Fixes #12`.
3. Explain what changed and how it was tested.
4. Ask a teammate to review.
5. Make requested updates, commit, and push again.
6. Merge only after approval and passing checks.

---

## 5. Use GitHub Projects

Create a project board with these columns:

**Todo → In progress → Review → Done**

- Add Issues to the board.
- Assign each task to a person.
- Add labels such as `data`, `analysis`, `documentation`.
- Add due dates and priorities if useful.
- Move the Issue when its status changes.

The Project shows **what the team is doing**; the repository contains **the actual files**.

---

## 6. Example for our team

**Issue:** Clean missing laboratory values

1. Add the Issue to the Project: **Todo**.
2. Create branch: `feature/clean-lab-data`.
3. Make and commit the changes.
4. Push the branch and open a PR.
5. Move the Issue to **Review**.
6. After approval and merge, move it to **Done**.

---

## 7. Team rules before Monday

- Pull `main` before starting work.
- Use clear branch and commit names.
- Keep PRs small and focused.
- Review someone else's PR.
- Do not commit passwords, patient identifiers, or other sensitive data.
- Confirm the Project board reflects the real status.

**Remember:** Track the task in Projects, make changes in a branch, and merge through a reviewed PR.
