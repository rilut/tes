from keras.optimizers import Adam
import keras.backend as K

class Optimizer_1(Adam):
    ''''''
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms

        for p, g, m in zip(params, grads, ms):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            clipvalue = 0.0001
            p_t = p - (K.exp(K.sign(g) * K.sign(m_t)) + K.clip(g, -clipvalue, clipvalue) * g)

            self.updates.append((m, m_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates


class Optimizer_2(Adam):
    ''''''
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms

        for p, g, m in zip(params, grads, ms):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            p_t = p - lr_t * K.dropout(m_t, 0.3) * K.exp(0.001 * p)

            self.updates.append((m, m_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates
