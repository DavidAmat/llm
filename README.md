# llm

## Submodules management

To create a repository on GitHub that includes multiple forked repositories as primary folders alongside your own development folder, while allowing you to easily fetch updates from the original repositories, follow these steps:

### Step 1: **Add Forked Repositories as Git Submodules**
1. Inside the cloned repository folder, add each forked repository as a submodule:
   ```bash
   git submodule add git@github.com:DavidAmat/Hands-On-Large-Language-Models.git
   ```

2. Initialize and update the submodules:
   ```bash
   git submodule update --init --recursive
   ```

---

### Step 2: **Add Your Development Folder**
1. Create a new folder for your developments:
   ```bash
   mkdir david
   ```
2. Add your own Python scripts, modules, or other files here.

3. Stage and commit the changes:
   ```bash
   git add .
   git commit -m "Add forked repos and development folder"
   git push origin main
   ```

---

### Step 3: **Configure Submodules for Easy Updates**
1. To fetch updates from the original repositories, configure the submodules to track their remotes. Navigate into each submodule folder and set the upstream:
   ```bash
   cd forked-repo-1
   git remote add upstream https://github.com/originalowner/original-repo-1.git
   cd ..
   ```

   Repeat for each submodule.

2. To update all submodules, you can run:
   ```bash
   git submodule foreach 'git fetch upstream && git merge upstream/main'
   ```

---

### Step 4: **Push Your Changes**
1. Ensure all submodule references are committed:
   ```bash
   git add .
   git commit -m "Update submodule references"
   git push origin main
   ```

---

### Step 5: **Workflow for Future Updates**
1. To fetch updates for forked repositories:
   ```bash
   git submodule update --remote --merge
   ```

2. To commit your development changes:
   - Navigate to the `david` folder and make changes.
   - Stage and commit changes as usual:
     ```bash
     git add david/
     git commit -m "Your changes"
     git push origin main
     ```

---

### Notes:
- Submodules ensure that you can fetch updates without affecting your main repository.
- Your `david` folder is independent and will not be impacted by updates to the submodules.
- Use the `git submodule update` commands regularly to keep forked repositories in sync with their upstream sources.

## Remove a submodule

```bash
# Remove submodule entry from .gitmodules
git add .gitmodules

# Remove submodule configuration
git config -f .git/config --remove-section submodule.forked-repo-1

# Remove submodule folder and unstage
git rm -r forked-repo-1

# Commit the changes
git commit -m "Remove submodule forked-repo-1"

# Push the changes to the remote repository
git push origin master

# Optional: Clean up .git/modules directory
rm -rf .git/modules/Hands-On-Large-Language-Models
```

## Install virtual environments

### Book: Hands-On-Large-Language-Models

Install requirements:

```bash
pyenv install 3.10.14
pyenv local 3.10.14

# Notation: b_ -> book // hollm: hands on llm
conda create --name b_hollm python=3.10.14
conda activate b_hollm
# conda env update --file environment.yml
pip install -r requirements.txt
```

Install ipykernel

```bash
python -m ipykernel install --user --name "kr_hollm"
```