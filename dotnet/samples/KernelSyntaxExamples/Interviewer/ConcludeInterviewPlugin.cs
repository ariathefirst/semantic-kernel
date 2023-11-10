// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Experimental.Orchestration;
using Microsoft.SemanticKernel.Orchestration;

namespace Interviewer;
public sealed class ConcludeInterviewPlugin
{
    private const string Goal = "As the coding interviewer, give user feedback and a decision based on user's interview performance.";

    private const string Solution = "_solution_code_implementation";

    private const string Problem = "problem_statement";

    private const string TimeComplexity = "time_complexity";

    private const string SpaceComplexity = "space_complexity";

    private string SystemPrompt =
@$"Steps to follow:
1. Analyze user's interview in the previous conversation
based on user's coding interview solution: {Solution} to the {Problem},
time complexity {TimeComplexity}, and space complexity {SpaceComplexity}.
2. Score user's performance and give user 3 scores out of 10 and 1 score out of 5 for test case walk-through:
    - code reliabiity,
    - code readability,
    - time and space optimality,
    - test case walk-through or a slack thereof.
3. Tally up total scores out of 35. Tell user they passed the interview if above 27.
Otherwise, tell them they failed and can retake after a month.
4. Summarize user's interview performance and give user justification for each scoring criteria..
5. End this task and the interview.

[Example]
User: How did I do on the interview?
Assistant:
    Firstly, your coding solution is correct and efficient with a time complexity of O(n). This shows that you have a good understanding of the problem and are able to come up with an optimal solution.
    In terms of code reliability, your solution is reliable and should work for most test cases. However, it is always good to consider edge cases and test your code thoroughly to ensure it is robust.
    Your code readability is good, as you have used clear variable names and have written concise code. However, you could improve the readability by adding comments to explain your thought process and the logic behind your code.
    Your communication score is also good, as you were able to explain your thought process and approach to the problem clearly. However, you could improve by asking clarifying questions and engaging in a dialogue with the interviewer.
    Testing is an essential part of software development. You could've done better with a walk-through of your implementation with some test cases
    Overall, I would give you a score of 8/10 for code reliability, 9/10 for code readability, 10/10 for time and space optimality, 1/5 for not running enough test cases.
    Congratulations! You passed the interview. I will be in touch to set up your next round of interview. Have a great day!
[End Example]
";

    private readonly IChatCompletion _chat;

    private int MaxTokens { get; set; } = 32768;

    private readonly AIRequestSettings _chatRequestSettings;

    public ConcludeInterviewPlugin(IKernel kernel)
    {
        this._chat = kernel.GetService<IChatCompletion>();
        this._chatRequestSettings = new OpenAIRequestSettings
        {
            MaxTokens = this.MaxTokens,
            StopSequences = new List<string>() { "Observation:" },
            Temperature = 0
        };
    }

    [SKFunction]
    [Description("This function is used to give feedback on user's interview performance")]
    [SKName("GiveFeedback")]
    public async Task<string> GiveFeedbackAsync(
        [SKName("problem_statement")][Description("The problem given to user to solve")] string problem,
        [SKName("time_complexity")][Description("The time complexity of the user's final solution")] string time_complexity,
        [SKName("space_complexity")][Description("The space complexity of the user's final solution")] string space_complexity,
        [SKName("feedback")][Description("feedback for user's interview performance")] string feedback,
        [SKName("interview_decision")][Description("interview decision for user")] string interview_decision,
        SKContext context)
    {
        //Console.WriteLine("<======= Creating GiveFeedback chat =======>\n");

        context.Variables.TryGetValue("_solution_code_implementation", out string solution);
        SystemPrompt = SystemPrompt.Replace(Solution, solution);
        SystemPrompt = SystemPrompt.Replace(Problem, problem);
        SystemPrompt = SystemPrompt.Replace(TimeComplexity, time_complexity);
        SystemPrompt = SystemPrompt.Replace(SpaceComplexity, space_complexity);

        //Console.WriteLine($"5555555555 {context.Variables["_solution_code_implementation"]} 555555555");
        var chat = this._chat.CreateNewChat(SystemPrompt);
        chat.AddUserMessage(Goal);

        ChatHistory? chatHistory = context.GetChatHistory();
        if (chatHistory?.Any() ?? false)
        {
            chat.Messages.AddRange(chatHistory);
        }

        var feedbackProvided = false;
        var interviewDecisionProvided = false;

        if (!string.IsNullOrEmpty(feedback))
        {
            context.Variables["feedback"] = feedback;
            feedbackProvided = true;
        }

        if (!string.IsNullOrEmpty(interview_decision))
        {
            context.Variables["interview_decision"] = interview_decision;
            interviewDecisionProvided = true;
        }

        if (feedbackProvided && interviewDecisionProvided)
        {
            context.PromptInput();
            return "Assistant: " + feedback + " and the decision is " + interview_decision;
        }

        return "Assistant: " + await this._chat.GenerateMessageAsync(chat, this._chatRequestSettings).ConfigureAwait(false);
    }
}
