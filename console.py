from collections import deque

import textattack
import tqdm
import time

from search_method import OLMWordSwap

class Console:
    def __init__(self, data, num_examples, attack):
        self.data = data
        self.num_examples = num_examples
        self.attack = attack

    def print_console(self):
        num_remaining_attacks = self.num_examples
        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)

        worklist = deque(range(0, self.num_examples))
        worklist_tail = worklist[-1]

        attack_log_manager = textattack.loggers.AttackLogManager()

        load_time = time.time()
        
        num_results = 0
        num_failures = 0
        num_successes = 0
        
        for result in self.attack.attack_dataset(self.data, indices=worklist):
            print(result.__str__(color_method="ansi") + "\n")
            attack_log_manager.log_result(result)
            
            if not isinstance(result, textattack.attack_results.SkippedAttackResult):
                pbar.update(1)
            else:
                worklist_tail += 1
                pbar.update(1)
                worklist.append(worklist_tail)
                OLMWordSwap.index += 1

            num_results += 1

            if (
                type(result) == textattack.attack_results.SuccessfulAttackResult
                or type(result) == textattack.attack_results.MaximizedAttackResult
            ):
                num_successes += 1
                
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description(
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )

        pbar.close()
        print()

        attack_log_manager.enable_stdout()
        attack_log_manager.log_summary()
        attack_log_manager.flush()
        print()
        
        textattack.shared.logger.info(f"Attack time: {time.time() - load_time}s")
