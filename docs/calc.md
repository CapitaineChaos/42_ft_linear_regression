Dans le cas d’une **régression linéaire à une seule variable**.

- $x_i$ = la $i$-ème entrée du jeu de données
- $y_i$ = la vraie valeur associée
- $\theta_0,\theta_1$ = les paramètres du modèle
- $h_\theta(x_i)$ = la **prédiction** du modèle pour $x_i$

Autrement dit, $h_\theta$ est juste une fonction. On la note ainsi pour dire : “la fonction $h$, paramétrée par $\theta$”.  
Ici :

$$
h_\theta(x)=\theta_0+\theta_1 x
$$

Donc pour chaque donnée $x_i$, on calcule :

$$
h_\theta(x_i)=\theta_0+\theta_1 x_i
$$

C’est simplement la formule d’une droite.

---

## 1. Fonction de coût $J(\theta_0,\theta_1)$

On aurait pu choisir l'erreur absolue :

$$
J(\theta)=\frac{1}{m}\sum |h_\theta(x_i)-y_i|
$$

Elle est plus robuste aux grosses valeurs aberrantes, mais moins confortable pour la dérivation.

On choisira plutôt la MSE (Mean Squared Error), une forme de pénalité quadratique

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$

Elle mesure l’erreur moyenne quadratique, avec le facteur $1/2$ pour simplifier la dérivation.

---

## 2. Ce qu’on cherche exactement

On veut calculer :

$$
\frac{\partial J}{\partial \theta_0}
\qquad \text{et} \qquad
\frac{\partial J}{\partial \theta_1}
$$

En dérivée partielle :

- pour $\frac{\partial J}{\partial \theta_0}$, on considère $\theta_1$ comme constant ;
- pour $\frac{\partial J}{\partial \theta_1}$, on considère $\theta_0$ comme constant.

---

## 3. Utilisationz mathématique

### a) Linéarité de la dérivée

$$
\frac{d}{d\theta}\left(\sum_i f_i(\theta)\right)=\sum_i \frac{d}{d\theta}f_i(\theta)
$$

et aussi

$$
\frac{d}{d\theta}(c f)=c \frac{d}{d\theta}f
$$

si $c$ est une constante.

### b) Dérivée d’un carré

$$
\frac{d}{d\theta}(u(\theta)^2)=2u(\theta)\,u'(\theta)
$$

C’est la **règle de la chaîne** (fonction composée).

### c) La dérivée d’une expression affine

Si

$$
u(\theta_0,\theta_1)=\theta_0+\theta_1 x_i-y_i
$$

alors :

- par rapport à $\theta_0$,

  $$
  \frac{\partial u}{\partial \theta_0}=1
  $$

- par rapport à $\theta_1$,

  $$
  \frac{\partial u}{\partial \theta_1}=x_i
  $$

car $x_i$ et $y_i$ sont des constantes du dataset.

---

# 4. Dérivation pas à pas de $\partial J/\partial \theta_0$

On part de :

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$

Comme

$$
h_\theta(x_i)=\theta_0+\theta_1 x_i
$$

on peut écrire :

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1 x_i-y_i)^2
$$

Maintenant on dérive par rapport à $\theta_0$ :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{\partial}{\partial \theta_0}
\left(
\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1 x_i-y_i)^2
\right)
$$

On sort la constante $1/(2m)$ :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{2m}
\sum_{i=1}^{m}
\frac{\partial}{\partial \theta_0}
(\theta_0+\theta_1 x_i-y_i)^2
$$

Maintenant règle de la chaîne :

si

$$
u_i=\theta_0+\theta_1 x_i-y_i
$$

alors

$$
\frac{\partial}{\partial \theta_0}(u_i^2)=2u_i \frac{\partial u_i}{\partial \theta_0}
$$

Donc :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{2m}
\sum_{i=1}^{m}
2(\theta_0+\theta_1 x_i-y_i)\cdot 1
$$

car

$$
\frac{\partial}{\partial \theta_0}(\theta_0+\theta_1 x_i-y_i)=1
$$

On simplifie le $2$ avec le $1/2$ :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{m}
\sum_{i=1}^{m}
(\theta_0+\theta_1 x_i-y_i)
$$

et comme

$$
\theta_0+\theta_1 x_i = h_\theta(x_i)
$$

on obtient :

$$
\boxed{
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)
}
$$

---

# 5. Dérivation pas à pas de $\partial J/\partial \theta_1$

Même départ :

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1 x_i-y_i)^2
$$

On dérive par rapport à $\theta_1$ :

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{\partial}{\partial \theta_1}
\left(
\frac{1}{2m}\sum_{i=1}^{m}(\theta_0+\theta_1 x_i-y_i)^2
\right)
$$

On sort la constante et on dérive terme à terme :

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{2m}
\sum_{i=1}^{m}
\frac{\partial}{\partial \theta_1}
(\theta_0+\theta_1 x_i-y_i)^2
$$

Règle de la chaîne :

$$
\frac{\partial}{\partial \theta_1}(u_i^2)=2u_i \frac{\partial u_i}{\partial \theta_1}
$$

avec

$$
u_i=\theta_0+\theta_1 x_i-y_i
$$

Donc :

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{2m}
\sum_{i=1}^{m}
2(\theta_0+\theta_1 x_i-y_i)\cdot x_i
$$

car

$$
\frac{\partial}{\partial \theta_1}(\theta_0+\theta_1 x_i-y_i)=x_i
$$

On simplifie :

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{m}
\sum_{i=1}^{m}
(\theta_0+\theta_1 x_i-y_i)x_i
$$

et donc :

$$
\boxed{
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i
}
$$

---

# 6. Précision : pour $\theta_0$ on obtient $1$, et pour $\theta_1$ on obtient $x_i$

On part de :

$$
\theta_0+\theta_1 x_i-y_i
$$

### Dérivée par rapport à $\theta_0$

- $\frac{\partial \theta_0}{\partial \theta_0}=1$
- $\theta_1 x_i$ est constant par rapport à $\theta_0$, donc dérivée $=0$
- $y_i$ est constant, donc dérivée $=0$

Donc :

$$
\frac{\partial}{\partial \theta_0}(\theta_0+\theta_1 x_i-y_i)=1
$$

### Dérivée par rapport à $\theta_1$

- $\theta_0$ est constant, donc dérivée $=0$
- $\frac{\partial}{\partial \theta_1}(\theta_1 x_i)=x_i$
- $y_i$ est constant, donc dérivée $=0$

Donc :

$$
\frac{\partial}{\partial \theta_1}(\theta_0+\theta_1 x_i-y_i)=x_i
$$

---

# 7. Intuitivement

Le terme

$$
h_\theta(x_i)-y_i
$$

est l’**erreur** sur le point $i$.

Donc :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{m}\sum (erreur)
$$

c’est en gros la moyenne des erreurs.

Et

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{m}\sum (erreur)\cdot x_i
$$

c’est la moyenne des erreurs, mais pondérée par la valeur de $x_i$.  
C’est logique : $\theta_1$ contrôle la pente, donc son influence dépend de la position en $x$.

---

# 8. Forme simplifiée

Tu peux retenir cette chaîne :

$$
J=\frac{1}{2m}\sum (\text{prédiction} - \text{réel})^2
$$

puis :

$$
\frac{\partial J}{\partial \theta}
=
\frac{1}{m}\sum (\text{prédiction} - \text{réel}) \cdot \frac{\partial(\text{prédiction})}{\partial \theta}
$$

Or ici :

- prédiction $= h_\theta(x_i)=\theta_0+\theta_1 x_i$
- $\frac{\partial h_\theta(x_i)}{\partial \theta_0}=1$
- $\frac{\partial h_\theta(x_i)}{\partial \theta_1}=x_i$

donc :

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{m}\sum (h_\theta(x_i)-y_i)\cdot 1
$$

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{m}\sum (h_\theta(x_i)-y_i)\cdot x_i
$$

---

