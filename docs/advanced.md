# Courbes de niveau de la fonction de coût et trajectoire de la descente de gradient

## Sommaire

- [1. Objet du document](#s1)
- [2. Cadre mathématique](#s2)
- [3. Lexique des termes employés](#s3)
- [4. Lexique des symboles](#s4)
- [5. Modèle de régression linéaire simple](#s5)
- [6. Erreur ponctuelle, MSE et fonction de coût](#s6)
- [7. Interprétation géométrique dans l’espace des paramètres](#s7)
- [8. Développement complet de la fonction de coût](#s8)
- [9. Gradient de la fonction de coût](#s9)
- [10. Mini-cours sur la matrice hessienne](#s10)
- [11. Forme des courbes de niveau](#s11)
- [12. Minimum global et point optimal](#s12)
- [13. Descente de gradient](#s13)
- [14. Sens géométrique de la trajectoire](#s14)
- [15. Construction d’un contour plot](#s15)
- [16. Rôle du logarithme dans la représentation](#s16)
- [17. Effet de la normalisation des données](#s17)
- [18. Dé-normalisation des paramètres](#s18)
- [19. Lecture scientifique du graphique obtenu](#s19)
- [20. Formules à retenir](#s20)

---

<a id="s1"></a>
## 1. Objet du document

Présenter, dans le cadre de la régression linéaire simple, la construction mathématique d’un **contour plot** de la fonction de coût, puis l’interprétation de la **trajectoire de la descente de gradient** dans l’espace des paramètres.

Le graphique étudié ne représente pas les points du jeu de données dans le plan usuel $(x,y)$. Il représente la valeur de l’erreur globale du modèle dans le plan des paramètres $(\theta_0,\theta_1)$.

---

<a id="s2"></a>
## 2. Cadre mathématique

Considérer un ensemble de données

$$
\mathcal D = \{(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)\}.
$$

La variable $x$ désigne la variable explicative. La variable $y$ désigne la variable à prédire.

Le modèle linéaire étudié possède deux paramètres :

$$
\theta_0 \quad \text{et} \quad \theta_1.
$$

Le but de l’apprentissage consiste à déterminer des valeurs de $\theta_0$ et $\theta_1$ qui rendent le modèle aussi précis que possible sur les données disponibles.

---

<a id="s3"></a>
## 3. Lexique des termes employés

| Terme | Définition |
|---|---|
| Régression linéaire simple | Méthode de modélisation dans laquelle la variable prédite est approchée par une fonction affine d’une seule variable explicative. |
| Paramètre | Quantité fixée par l’apprentissage et non connue à l’avance. Ici, $\theta_0$ et $\theta_1$ sont les paramètres du modèle. |
| Ordonnée à l’origine | Valeur prédite lorsque $x=0$. Elle correspond à $\theta_0$. |
| Pente | Variation de la prédiction lorsque $x$ augmente d’une unité. Elle correspond à $\theta_1$. |
| Prédiction | Valeur produite par le modèle pour une observation donnée. |
| Erreur ponctuelle | Différence entre la valeur prédite et la valeur observée pour une observation. |
| Fonction de coût | Fonction qui associe à chaque couple de paramètres une erreur globale sur l’ensemble des données. |
| MSE | Abréviation de *Mean Squared Error*, c’est-à-dire moyenne des erreurs quadratiques. |
| Gradient | Vecteur formé des dérivées partielles de la fonction de coût. Il indique localement la direction de croissance la plus rapide. |
| Hessienne | Matrice des dérivées partielles secondes. Elle décrit la courbure locale de la fonction. |
| Courbe de niveau | Ensemble des points pour lesquels une fonction prend une valeur constante. |
| Contour plot | Représentation plane d’une fonction de deux variables à l’aide de courbes de niveau. |
| Minimum global | Point où la fonction atteint sa plus petite valeur sur tout son domaine. |
| Descente de gradient | Méthode itérative d’optimisation consistant à se déplacer dans la direction opposée au gradient. |
| Learning rate | Coefficient positif qui règle la taille du déplacement à chaque itération de la descente de gradient. |
| Convexité | Propriété géométrique assurant, dans ce contexte, l’absence de minima locaux parasites et l’existence d’un unique minimum global lorsque la hessienne est définie positive. |
| Normalisation | Transformation des données visant à ramener les valeurs dans une échelle plus homogène. |

---

<a id="s4"></a>
## 4. Lexique des symboles

| Symbole | Définition |
|---|---|
| $\mathcal{D}$ | Ensemble des observations. |
| $m$ | Nombre total d’observations. |
| $(x_i, y_i)$ | $i$-ième observation. |
| $\theta_0$ | Ordonnée à l’origine du modèle. |
| $\theta_1$ | Pente du modèle. |
| $\theta = (\theta_0, \theta_1)$ | Vecteur des paramètres. |
| $h_\theta(x)$ | Prédiction du modèle pour l’entrée $x$. |
| $\hat{y}_i$ | Valeur prédite pour la $i$-ième observation. |
| $e_i$ | Erreur ponctuelle pour la $i$-ième observation. |
| $J(\theta_0, \theta_1)$ | Fonction de coût. |
| $\nabla J$ | Gradient de la fonction $J$. |
| $H$ | Matrice hessienne de $J$. |
| $\eta$ | Learning rate. |
| $\bar{x} = \frac{1}{m}\sum_{i=1}^m x_i$ | Moyenne des valeurs de $x$. |
| $\bar{y} = \frac{1}{m}\sum_{i=1}^m y_i$ | Moyenne des valeurs de $y$. |
| $\overline{x^2} = \frac{1}{m}\sum_{i=1}^m x_i^2$ | Moyenne des carrés des valeurs de $x$. |
| $\overline{xy} = \frac{1}{m}\sum_{i=1}^m x_i y_i$ | Moyenne des produits $x_i y_i$. |
| $(\theta_0^\star, \theta_1^\star)$ | Couple optimal — minimum global de la fonction de coût ($\star$ = optimal). |
| $\mathbb{R}$ | Ensemble des nombres réels. |
| $\mathbb{R}_{>0}$ | Ensemble des réels strictement positifs. |

---

<a id="s5"></a>
## 5. Modèle de régression linéaire simple

Le modèle affine étudié est

$$
h_\theta(x)=\theta_0+\theta_1 x.
$$

Pour l’observation $(x_i,y_i)$, la valeur prédite vaut

$$
\hat y_i = h_\theta(x_i)=\theta_0+\theta_1 x_i.
$$

Le paramètre $\theta_0$ règle le décalage vertical de la droite. Le paramètre $\theta_1$ règle l’inclinaison de la droite.

---

<a id="s6"></a>
## 6. Erreur ponctuelle, MSE et fonction de coût

### 6.1. Erreur ponctuelle

Pour chaque observation, définir l’erreur

$$
e_i = h_\theta(x_i)-y_i = \theta_0+\theta_1 x_i-y_i.
$$

Cette erreur peut être positive, négative ou nulle.

### 6.2. Erreur quadratique moyenne

Afin d’éviter la compensation entre erreurs positives et erreurs négatives, considérer les carrés des erreurs :

$$
e_i^2 = (\theta_0+\theta_1 x_i-y_i)^2.
$$

La moyenne de ces carrés est la MSE :

$$
\operatorname{MSE}(\theta_0,\theta_1)=\frac{1}{m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)^2.
$$

### 6.3. Fonction de coût

Introduire la fonction de coût usuelle :

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)^2.
$$

Le facteur $\frac{1}{2}$ simplifie les dérivées. Il ne modifie pas la position du minimum.

La relation entre MSE et fonction de coût est

$$
\operatorname{MSE}=2J.
$$

---
<a id="s7"></a>
## 7. Interprétation géométrique dans l'espace des paramètres

La fonction de coût dépend de deux variables : $\theta_0$ et $\theta_1$.

Il devient donc possible d’interpréter $J$ comme une surface au-dessus du plan des paramètres :

$$
(\theta_0,\theta_1) \longmapsto J(\theta_0,\theta_1).
$$

Chaque point du plan $(\theta_0,\theta_1)$ correspond à une droite candidate.

Chaque hauteur au-dessus de ce point correspond au coût associé à cette droite.

Le contour plot constitue la projection plane de cette surface, obtenue en traçant les courbes

$$
J(\theta_0,\theta_1)=c,
$$

pour différentes constantes $c$.

---

<a id="s8"></a>
## 8. Développement complet de la fonction de coût

Partir de

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)^2.
$$

Développer le carré :

$$
(\theta_0+\theta_1 x_i-y_i)^2
= \theta_0^2 + 2\theta_0\theta_1 x_i - 2\theta_0 y_i + \theta_1^2 x_i^2 - 2\theta_1 x_i y_i + y_i^2.
$$

Remplacer dans la somme :

$$
J(\theta_0,\theta_1)
=\frac{1}{2m}\sum_{i=1}^m \left(
\theta_0^2 + 2\theta_0\theta_1 x_i - 2\theta_0 y_i + \theta_1^2 x_i^2 - 2\theta_1 x_i y_i + y_i^2
\right).
$$

Utiliser la linéarité de la somme :

$$
J(\theta_0,\theta_1)
=\frac{1}{2m}\left(
\sum_{i=1}^m \theta_0^2
+ \sum_{i=1}^m 2\theta_0\theta_1 x_i
- \sum_{i=1}^m 2\theta_0 y_i
+ \sum_{i=1}^m \theta_1^2 x_i^2
- \sum_{i=1}^m 2\theta_1 x_i y_i
+ \sum_{i=1}^m y_i^2
\right).
$$

Comme $\theta_0$ et $\theta_1$ ne dépendent pas de l’indice $i$, simplifier :

$$
\sum_{i=1}^m \theta_0^2 = m\theta_0^2,
$$

$$
\sum_{i=1}^m 2\theta_0\theta_1 x_i = 2\theta_0\theta_1 \sum_{i=1}^m x_i,
$$

$$
\sum_{i=1}^m 2\theta_0 y_i = 2\theta_0 \sum_{i=1}^m y_i,
$$

$$
\sum_{i=1}^m \theta_1^2 x_i^2 = \theta_1^2 \sum_{i=1}^m x_i^2,
$$

$$
\sum_{i=1}^m 2\theta_1 x_i y_i = 2\theta_1 \sum_{i=1}^m x_i y_i.
$$

Introduire les moyennes usuelles :

$$
\bar x = \frac{1}{m}\sum_{i=1}^m x_i,
\qquad
\bar y = \frac{1}{m}\sum_{i=1}^m y_i,
$$

$$
\overline{x^2} = \frac{1}{m}\sum_{i=1}^m x_i^2,
\qquad
\overline{xy} = \frac{1}{m}\sum_{i=1}^m x_i y_i,
\qquad
\overline{y^2} = \frac{1}{m}\sum_{i=1}^m y_i^2.
$$

Obtenir finalement

$$
J(\theta_0,\theta_1)
= \frac{1}{2}\theta_0^2
+ \theta_0\theta_1\bar x
- \theta_0\bar y
+ \frac{1}{2}\theta_1^2\overline{x^2}
- \theta_1\overline{xy}
+ \frac{1}{2}\overline{y^2}.
$$

Cette écriture montre explicitement que $J$ est un polynôme de degré $2$ en $(\theta_0,\theta_1)$.

---

<a id="s9"></a>
## 9. Gradient de la fonction de coût

Le gradient est le vecteur des dérivées partielles premières.

### 9.1. Dérivée partielle par rapport à $\theta_0$

À partir de

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)^2,
$$

obtenir

$$
\frac{\partial J}{\partial \theta_0}
=\frac{1}{2m}\sum_{i=1}^m 2(\theta_0+\theta_1 x_i-y_i)
=\frac{1}{m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i).
$$

### 9.2. Dérivée partielle par rapport à $\theta_1$

De même,

$$
\frac{\partial J}{\partial \theta_1}
=\frac{1}{2m}\sum_{i=1}^m 2(\theta_0+\theta_1 x_i-y_i)x_i
=\frac{1}{m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)x_i.
$$

### 9.3. Écriture vectorielle du gradient

Ainsi,

$$
\nabla J(\theta_0,\theta_1)
=
\left(
\frac{\partial J}{\partial \theta_0},
\frac{\partial J}{\partial \theta_1}
\right).
$$

Le gradient indique la direction locale de croissance la plus rapide de la fonction. La direction opposée correspond donc à une diminution locale de la fonction de coût.

---

<a id="s10"></a>
## 10. Mini-cours sur la matrice hessienne

### 10.1. Définition générale

Pour une fonction de deux variables $f(u,v)$, la matrice hessienne est

$$
H_f(u,v)=
\begin{pmatrix}
\dfrac{\partial^2 f}{\partial u^2} & \dfrac{\partial^2 f}{\partial u\partial v} \\
\dfrac{\partial^2 f}{\partial v\partial u} & \dfrac{\partial^2 f}{\partial v^2}
\end{pmatrix}.
$$

Elle rassemble toutes les dérivées partielles secondes. Elle décrit la courbure locale de la fonction.

### 10.2. Signification des dérivées secondes

- $\dfrac{\partial^2 f}{\partial u^2}$ mesure la courbure selon la direction $u$ ;
- $\dfrac{\partial^2 f}{\partial v^2}$ mesure la courbure selon la direction $v$ ;
- $\dfrac{\partial^2 f}{\partial u\partial v}$ mesure le couplage entre les deux directions.

### 10.3. Hessienne de la fonction de coût

À partir des dérivées premières, calculer les dérivées secondes :

$$
\frac{\partial^2 J}{\partial \theta_0^2}=1,
$$

$$
\frac{\partial^2 J}{\partial \theta_0\partial \theta_1}=\bar x,
$$

$$
\frac{\partial^2 J}{\partial \theta_1\partial \theta_0}=\bar x,
$$

$$
\frac{\partial^2 J}{\partial \theta_1^2}=\overline{x^2}.
$$

Donc,

$$
H=
\begin{pmatrix}
1 & \bar x \\
\bar x & \overline{x^2}
\end{pmatrix}.
$$

### 10.4. Symétrie de la matrice

Les dérivées croisées sont égales :

$$
\frac{\partial^2 J}{\partial \theta_0\partial \theta_1}
=
\frac{\partial^2 J}{\partial \theta_1\partial \theta_0}.
$$

Cette propriété est classique pour les fonctions suffisamment régulières.

### 10.5. Rôle de la hessienne dans l’étude géométrique

La hessienne permet de répondre à trois questions essentielles :

1. la fonction est-elle convexe ;
2. les courbes de niveau sont-elles circulaires ou elliptiques ;
3. l’orientation et l’allongement des courbes dépendent-ils d’un couplage entre les variables.

### 10.6. Condition de convexité

Une matrice symétrique réelle est définie positive lorsque toutes ses directions présentent une courbure strictement positive.

Pour une matrice $2 \times 2$,

$$
\begin{pmatrix}
a & b \\
b & d
\end{pmatrix},
$$

une condition pratique de définition positive est

$$
a>0
\qquad \text{et} \qquad
ad-b^2>0.
$$

Ici,

$$
a=1,
\qquad b=\bar x,
\qquad d=\overline{x^2}.
$$

Le déterminant vaut

$$
\det(H)=\overline{x^2}-\bar x^2.
$$

Or

$$
\overline{x^2}-\bar x^2
$$

est la variance empirique de $x$. Si les valeurs de $x$ ne sont pas toutes identiques, cette variance est strictement positive. Dans ce cas, la hessienne est définie positive.

### 10.7. Conséquence sur l’optimisation

Si la hessienne est définie positive, la fonction de coût est strictement convexe. Elle possède alors un unique minimum global.

---

<a id="s11"></a>
## 11. Forme des courbes de niveau

Considérer une équation de niveau

$$
J(\theta_0,\theta_1)=c.
$$

Comme $J$ est une fonction quadratique en $(\theta_0,\theta_1)$, cette équation est une équation quadratique à deux variables. Dans le cas convexe précédent, elle décrit une ellipse.

Plus précisément, une fonction quadratique de la forme

$$
q(u,v)=\frac{1}{2}
\begin{pmatrix}u & v\end{pmatrix}
A
\begin{pmatrix}u \\ v\end{pmatrix}
+ \text{termes affines} + \text{constante},
$$

avec $A$ symétrique définie positive, possède des courbes de niveau elliptiques concentriques autour du point optimal $(\theta_0^\star,\theta_1^\star)$.

---
<a id="s12"></a>
## 12. Minimum global et point optimal

Le point optimal est le couple $(\theta_0^\star,\theta_1^\star)$ où la fonction de coût est minimale.

Ce point vérifie

$$
\nabla J(\theta_0^\star,\theta_1^\star)=0.
$$

Comme la fonction est strictement convexe lorsque les $x_i$ ne sont pas tous égaux, ce point critique est l’unique minimum global.

Toutes les courbes de niveau sont donc centrées autour de ce point.

---
<a id="s13"></a>
## 13. Descente de gradient

### 13.1. Principe

La descente de gradient construit une suite de paramètres

$$
(\theta_0^{(0)},\theta_1^{(0)}),
(\theta_0^{(1)},\theta_1^{(1)}),
\dots,
(\theta_0^{(k)},\theta_1^{(k)}),
\dots
$$

par la règle de mise à jour

$$
\theta_0^{(k+1)} = \theta_0^{(k)} - \eta\,\frac{\partial J}{\partial \theta_0}(\theta_0^{(k)},\theta_1^{(k)}),
$$

$$
\theta_1^{(k+1)} = \theta_1^{(k)} - \eta\,\frac{\partial J}{\partial \theta_1}(\theta_0^{(k)},\theta_1^{(k)}).
$$

### 13.2. Forme explicite

En remplaçant les dérivées par leurs expressions,

$$
\theta_0^{(k+1)}
= \theta_0^{(k)}-
\eta\left[
\frac{1}{m}\sum_{i=1}^m \big(\theta_0^{(k)}+\theta_1^{(k)}x_i-y_i\big)
\right],
$$

$$
\theta_1^{(k+1)}
= \theta_1^{(k)}-
\eta\left[
\frac{1}{m}\sum_{i=1}^m \big(\theta_0^{(k)}+\theta_1^{(k)}x_i-y_i\big)x_i
\right].
$$

### 13.3. Rôle du learning rate

Le paramètre $\eta$ règle l’amplitude du déplacement.

- si $\eta$ est trop petit, la convergence est lente ;
- si $\eta$ est convenable, la convergence est stable ;
- si $\eta$ est trop grand, la trajectoire peut osciller ou diverger.

---
<a id="s14"></a>
## 14. Sens géométrique de la trajectoire

Chaque itération de la descente de gradient fournit un nouveau point dans le plan $(\theta_0,\theta_1)$.

La suite de ces points constitue la trajectoire de l’algorithme.

Sur le contour plot :

- un point représente une droite candidate ;
- une courbe représente un ensemble de droites donnant le même coût ;
- la trajectoire montre comment l’algorithme se déplace d’une droite candidate à une autre jusqu’au voisinage du minimum global.

Dans une vallée très allongée, la trajectoire peut présenter un mouvement de zigzag.

---
<a id="s15"></a>
## 15. Construction d'un contour plot

La construction numérique suit les étapes suivantes.

### Étape 1

Choisir un domaine du plan des paramètres :

$$
\theta_0 \in [a,b],
\qquad
\theta_1 \in [c,d].
$$

### Étape 2

Construire une grille régulière de points dans ce rectangle.

### Étape 3

Évaluer la fonction de coût $J(\theta_0,\theta_1)$ en chaque point de la grille.

### Étape 4

Sélectionner plusieurs valeurs de niveau

$$
c_1,c_2,\dots,c_p.
$$

### Étape 5

Tracer les courbes

$$
J(\theta_0,\theta_1)=c_1,
\qquad
J(\theta_0,\theta_1)=c_2,
\qquad
\dots,
\qquad
J(\theta_0,\theta_1)=c_p.
$$

### Étape 6

Superposer les points successifs de la descente de gradient.

Remarque : le choix du nombre de niveaux est un choix de visualisation. La géométrie générale de la surface de coût n’est pas arbitraire.

---
<a id="s16"></a>
## 16. Rôle du logarithme dans la représentation

Il est fréquent de représenter non pas directement $J$, mais

$$
\log(J + \varepsilon),
$$

où $\varepsilon > 0$ est une très petite constante destinée à éviter le logarithme de zéro.

### 16.1. Justification mathématique

Le logarithme est strictement croissant sur $\mathbb{R}_{>0}$. Donc, si

$$
J_1 < J_2,
$$

alors

$$
\log(J_1+\varepsilon) < \log(J_2+\varepsilon).
$$

L’ordre des niveaux est conservé. Le minimum reste le minimum, et la structure générale des courbes de niveau est préservée.

### 16.2. Justification visuelle

Lorsque les coûts sont très dispersés, le logarithme compresse les grandes valeurs et étale davantage les petites valeurs. La zone proche du minimum devient alors plus lisible.

---
<a id="s17"></a>
## 17. Effet de la normalisation des données

Supposer les transformations

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}},
\qquad
y' = \frac{y-y_{\min}}{y_{\max}-y_{\min}}.
$$

La normalisation réduit les écarts d’échelle entre les variables.

Conséquences principales :

1. gradients généralement plus modérés ;
2. apprentissage plus stable ;
3. choix du learning rate plus simple ;
4. contour plot souvent plus lisible.

Sans normalisation, la vallée de la fonction de coût peut être extrêmement allongée, ce qui rend l’optimisation plus délicate.

---
<a id="s18"></a>
## 18. Dé-normalisation des paramètres

Dans l’espace normalisé, le modèle s’écrit

$$
y' = \theta_0' + \theta_1' x'.
$$

Remplacer $x'$ et $y'$ par leurs définitions :

$$
\frac{y-y_{\min}}{y_{\max}-y_{\min}}
=
\theta_0' + \theta_1'\frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

Multiplier par $y_{\max}-y_{\min}$ :

$$
y-y_{\min}
=
(y_{\max}-y_{\min})\theta_0'
+
(y_{\max}-y_{\min})\theta_1'\frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

Développer :

$$
y
=
y_{\min}
+
(y_{\max}-y_{\min})\theta_0'
-
(y_{\max}-y_{\min})\theta_1'\frac{x_{\min}}{x_{\max}-x_{\min}}
+
(y_{\max}-y_{\min})\theta_1'\frac{x}{x_{\max}-x_{\min}}.
$$

Comparer avec la forme

$$
y = \theta_0 + \theta_1 x.
$$

Obtenir alors

$$
\theta_1 = \theta_1'\frac{y_{\max}-y_{\min}}{x_{\max}-x_{\min}},
$$

$$
\theta_0 = y_{\min} + (y_{\max}-y_{\min})\theta_0' - \theta_1 x_{\min}.
$$

---
<a id="s19"></a>
## 19. Lecture scientifique du graphique obtenu

Le contour plot accompagné de la trajectoire de la descente de gradient permet d’observer simultanément :

1. la structure quadratique de la fonction de coût ;
2. la position du minimum global ;
3. la géométrie de la vallée de coût ;
4. la qualité du choix du learning rate ;
5. la stabilité ou l’instabilité de la convergence.

### 19.1. Learning rate faible

La trajectoire progresse lentement vers le minimum.

### 19.2. Learning rate adapté

La trajectoire converge de manière stable vers le centre des ellipses.

### 19.3. Learning rate excessif

La trajectoire oscille de part et d’autre de la vallée ou diverge hors de la zone de stabilité.

---
<a id="s20"></a>
## 20. Formules à retenir

### Modèle

$$
h_\theta(x)=\theta_0+\theta_1 x.
$$

### Fonction de coût

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)^2.
$$

### Gradient

$$
\frac{\partial J}{\partial \theta_0}
=\frac{1}{m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i),
$$

$$
\frac{\partial J}{\partial \theta_1}
=\frac{1}{m}\sum_{i=1}^m (\theta_0+\theta_1 x_i-y_i)x_i.
$$

### Hessienne

$$
H=
\begin{pmatrix}
1 & \bar x \\
\bar x & \overline{x^2}
\end{pmatrix}.
$$

### Mise à jour de la descente de gradient

$$
\theta_0^{(k+1)} = \theta_0^{(k)} - \eta\,\frac{\partial J}{\partial \theta_0}(\theta_0^{(k)},\theta_1^{(k)}),
$$

$$
\theta_1^{(k+1)} = \theta_1^{(k)} - \eta\,\frac{\partial J}{\partial \theta_1}(\theta_0^{(k)},\theta_1^{(k)}).
$$

### Point optimal

$$
\nabla J(\theta_0^\star,\theta_1^\star)=0.
$$

---

