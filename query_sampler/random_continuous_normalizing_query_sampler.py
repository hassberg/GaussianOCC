from active_learning_ts.query_selection.query_sampler import QuerySampler
import tensorflow as tf


def loop_body(iterator, body_index, body_total_covered, body_next_total, body_correct_offset, body_size_list,
              body_start_value):
    body_total_covered = body_next_total
    body_next_total += tf.gather(body_size_list, iterator)
    body_correct_offset = tf.gather(body_start_value, iterator)

    return iterator + 1, body_index, body_total_covered, body_next_total, body_correct_offset, body_size_list, body_start_value


class RandomContinuousNormalizingQuerySampler(QuerySampler):
    def sample(self, num_queries: int = 1) -> tf.Tensor:
        if self.pool.is_discrete():
            raise Exception("Should only be continuous pool")
        else:
            rd = tf.random.uniform(shape=(num_queries, self.pool.shape[0]))

            a = tf.convert_to_tensor([self.get_element_normalized(x) for x in rd])
            a = tf.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))

        return tf.convert_to_tensor(a)

    def update_pool(self, pool):
        self.pool = pool

    def get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        """
        The ranges are normalised by removing any gaps between ranges and then mapping the values onto the interval
        [0,1).
        This function un-normalizes a given vector.

        :param element: a tensor with entries in the range [0,1)
        :return: The un-normalized vector
        """
        # please do not try to read this code

        indices = tf.unstack(element)

        out = []

        # declare all variables at the beginning in order to avoid tf issues
        next_out = None
        start_value_list_iterator = iter(self.pool.start_values)
        size_list_iterator = iter(self.pool.sizes)
        total_size_iterator = iter(self.pool.total_sizes)
        total_covered = 0.0
        next_total = 0.0
        correct_offset = 0.0

        for index in indices:
            index = index * next(total_size_iterator)

            total_covered = 0.0
            next_total = 0.0
            correct_offset = 0.0
            start_value_list = next(start_value_list_iterator)
            size_list = next(size_list_iterator)

            j, i, total_covered, y, correct_offset, l1, l2 = tf.while_loop(
                lambda j, i, x, y, z, l1, l2: tf.math.less_equal(y, i),
                loop_body,
                [0, index, total_covered, next_total, correct_offset, size_list, start_value_list],
                parallel_iterations=1)

            # at this point, we have located the correct range, we just need to find the percentage of this range in
            # which the given point lies

            next_out = index - total_covered + correct_offset

            out.append(next_out)

        return tf.reshape(tf.stack(out), (1, len(out)))
