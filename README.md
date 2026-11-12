(using ruff for formatting)

#### HISTORY 
- I begun with a simple mlp implementation using the pytorch official tutorial as an example. Then I implemented a basic version of fsgm. these two initial experiments are currently untracked but can be found in the history of the commits. 
- From there since I had different attacks and mlp architectures to build and test I decided to start building a slightly more structured setup. I decided to implement attacks as objects which inherit from a base classis Attack, each of them has a function attack() which performs the attack over a model. 
- I implemented FSGM, IFSGM and PGD (Later I made this general, you can pass the kind of norm to use as a parameter when initializing, None means l-inf. This was done so that the Certified Accuracy can be used when working with l-2).
- In experiments/ I write code using the classes and function in src which are built to easily evaluate models over different attacks. The last thing I did is building a very rough version of an evaluation and plotting over a two moon database using a linear nn and an 1-Lipshitz one. 

The project is still in an embryonic version and there are so many things to change / improve.

#### TODO LIST
- Add model.eval() and model.train() where it is not done yet
- Check overfitting? 
- Add other metrics (even stuff related to how much time does training / evaluating etc take)
- Check random seeds
- Clean up the code, use more functions and find out if more suitable torch functions can substitute parts of the code
- Use a standard way to name trained models and maybe split experiments into training and evaluation?
- Add some testing? 
- Maybe use some AI tool to detect tricky bugs which didn't arise until now / to find out if there are possible optimization / more ideomatic way of writing the code 