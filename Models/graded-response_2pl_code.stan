
data {
  int<lower=2, upper=4> K; //number of categories
  int <lower=0> N_student;
  int <lower=0> N_item;
  int<lower=1, upper=K> Y[N_student, N_item];
}

parameters {
  vector[N_student] theta;
  real<lower=0> a[N_item]; // item discrimination
  real<lower=0> b[N_item]; // item difficulty
  ordered[K-1] kappa[N_item]; // category difficulty
  real mu_kappa; // mean of the prior distribution of category difficulty
  real<lower=0> sigma_kappa; // sd of the prior distribution of category difficulty
}

model {
  a ~ cauchy(0, 5);
  b ~ normal(0, 5);
  theta ~ normal(0, 1);
  
  for (n_item in 1:N_item) {
    for (k in 1:(K - 1)) {
      kappa[n_item, k] ~ normal(mu_kappa, sigma_kappa);
    }
  }
  
  mu_kappa ~ normal(0, 5);
  sigma_kappa ~ cauchy(0, 5);
  
  for (n_student in 1:N_student) {
    for (n_item in 1:N_item) {
      Y[n_student, n_item] ~ ordered_logistic((theta[n_student] - b[n_item]) * a[n_item], kappa[n_item]);
    }
  }
}
